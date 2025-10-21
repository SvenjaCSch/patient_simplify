import os
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain import PromptTemplate, LLMChain
import wandb
from textstat import flesch_reading_ease
import re
from dotenv import load_dotenv
import requests

class LangChain:
    def __init__(self):
        self.model_id = "KrithikV/MedMobile"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id)

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=200,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        self.hf = HuggingFacePipeline(pipeline=self.pipe)
        load_dotenv()
        self.promptlayer_api_key = os.getenv("PROMPTLAYER_API_KEY")
        os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY")
        wandb.init(project="patient_simplify")

    def generate_prompt_template(self, template_text=None):
        if template_text is None:
            template_text = """
        You are a medical communication assistant specialized in explaining complex medical information to patients in clear, simple, and empathetic language.
    
        Your tasks:
        - Simplify medical texts written by doctors.
        - Preserve the *medical accuracy* but remove unnecessary jargon.
        - Use neutral, supportive, and professional tone.
        - Structure explanations with short paragraphs, bullet points, and optional summaries.
        - Never add or invent medical information not in the original text.
        - Do not provide medical advice beyond what is in the original.
    
        Doctor's report:
        {doctor_text}
        """
        return PromptTemplate.from_template(template_text)

    def generate_chain(self, prompt_template):
        return LLMChain(prompt=prompt_template, llm=self.hf)


    def generate_summary(self, chain, doctor_text):
        summary = chain.run({"doctor_text": doctor_text})
        print(summary)
        return summary

    def log_metrics(self, summary, prompt_version):
        word_count = len(summary.split())
        sentence_count = len(re.findall(r'[.!?]', summary))
        reading_score = flesch_reading_ease(summary)

        wandb.log({
            "prompt_version": prompt_version,
            "summary_word_count": word_count,
            "summary_sentence_count": sentence_count,
            "flesch_score": reading_score,
            "summary_text": summary
        })

    def log_promptlayer(self, prompt_text, summary, prompt_version="default"):
        """
        Manually logs your MedMobile run to PromptLayer via REST API.
        """
        if not self.promptlayer_api_key:
            print("No PromptLayer API key found. Skipping logging.")
            return

        data = {
            "prompt": prompt_text,
            "response": summary,
            "model": self.model_id,
            "tags": [prompt_version],
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.promptlayer_api_key}"
        }

        try:
            res = requests.post("https://api.promptlayer.com/v1/prompts",
                                json=data, headers=headers)
            if res.ok:
                print(f"Logged to PromptLayer (tag: {prompt_version})")
            else:
                print(f"PromptLayer logging failed: {res.status_code} - {res.text}")
        except Exception as e:
            print(f"PromptLayer logging error: {e}")


    def select_best_summary(self, summaries_metrics, min_words=20):
        filtered = [s for s in summaries_metrics if s["word_count"] >= min_words]
        if not filtered:
            return None
        best = max(filtered, key=lambda x: x["flesch_score"])
        return best


    def manage_langchain(self, doctor_text, template_text=None, prompt_version="default"):
        prompt_template = self.generate_prompt_template(template_text)
        chain = self.generate_chain(prompt_template)
        summary = self.generate_summary(chain, doctor_text)
        self.log_promptlayer(prompt_template.template, summary, prompt_version)
        self.log_metrics(summary, prompt_version)
        return summary


if __name__ == "__main__":
    med_chain = LangChain()
    summaries_metrics = []
    template_v2 = "Explain this medical report in simple language for a patient (under 100 words):\n{doctor_text}"

    report1 = """
Patient shows elevated blood sugar (HbA1c 7.8%), slightly high LDL cholesterol, blood pressure in the upper normal range.
Recommendation: lifestyle modification, follow-up in 3 months.
"""
    report2 = """Patient has mild hypertension and elevated liver enzymes. Recommendation: diet change, regular exercise, monitor labs in 6 weeks."""

    reports = [report1, report2]
    templates = [("v1", None), ("v2", template_v2)]

    summaries_metrics = []

    for i, r in enumerate(reports, start=1):
        for version, tmpl in templates:
            prompt_tag = f"{version}_r{i}"
            summary = med_chain.manage_langchain(r, template_text=tmpl, prompt_version=prompt_tag)
            summaries_metrics.append({
                "prompt_version": prompt_tag,
                "summary": summary,
                "flesch_score": flesch_reading_ease(summary),
                "word_count": len(summary.split())
            })

    # Select best summary
    best_summary = med_chain.select_best_summary(summaries_metrics, min_words=20)

    print("\n=== Best Prompt Version ===")
    print(f"Version: {best_summary['prompt_version']}")
    print(f"Flesch Score: {best_summary['flesch_score']}")
    print(f"Word Count: {best_summary['word_count']}")
    print("Summary:\n", best_summary["summary"])