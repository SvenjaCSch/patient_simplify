from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain import PromptTemplate, LLMChain
import promptlayer
import wandb
from textstat import flesch_reading_ease
import re
from trulens_eval import Tru, Feedback
from promptlayer import PromptLayer

class LangChain:
    def __init__(self):
        self.model_id = "KrithikV/MedMobile"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id)

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=512,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        self.hf = HuggingFacePipeline(pipeline=self.pipe)
        wandb.init(project="patient_summary_demo")
        self.pl_client = PromptLayer()

    def generate_prompt_template(self, template_text=None):
        if template_text is None:
            template_text = (
                "Fasse den folgenden medizinischen Befund so zusammen, "
                "dass ein Patient ohne medizinische Vorkenntnisse ihn versteht:\n{befund}"
            )
        return PromptTemplate.from_template(template_text)

    def generate_chain(self, prompt):
        return LLMChain(prompt=prompt, llm=self.hf)

    def generate_example(self):
        return (
            "Patient zeigt erhöhte Blutzuckerwerte (HbA1c 7,8%), leicht erhöhtes LDL-Cholesterin, "
            "Blutdruck im oberen Normalbereich.\n"
            "Empfehlung: Ernährungsumstellung, regelmäßige Bewegung, Kontrolle in 3 Monaten, "
            "mögliche medikamentöse Therapie bei Verschlechterung."
        )

    def generate_summary(self, chain, findings):
        summary = chain.run(findings)
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
        self.pl_client.log_prompt(
            prompt_text=prompt_text,
            output_text=summary,
            tags=[prompt_version]
        )

    def select_best_summary(self, summaries_metrics, min_words=20):
        filtered = [s for s in summaries_metrics if s["word_count"] >= min_words]
        best = max(filtered, key=lambda x: x["flesch_score"])
        return best

    def manage_langchain(self, template_text=None, example_text=None, prompt_version="default"):
        prompt = self.generate_prompt_template(template_text)
        chain = self.generate_chain(prompt)
        example = example_text or self.generate_example()
        summary = self.generate_summary(chain, example)
        self.log_promptlayer(prompt.template, summary, prompt_version)
        self.log_metrics(summary, prompt_version)
        return summary

if __name__ == "__main__":
    med_chain = LangChain()
    summaries_metrics = []

    summary_v1 = med_chain.manage_langchain(prompt_version="v1")
    summaries_metrics.append({
        "prompt_version": "v1",
        "summary": summary_v1,
        "flesch_score": flesch_reading_ease(summary_v1),
        "word_count": len(summary_v1.split())
    })

    template_v2 = (
        "Erkläre diesen medizinischen Bericht in einfacher Sprache für einen Patienten. "
        "Halte die Zusammenfassung unter 100 Wörtern:\n{befund}"
    )
    summary_v2 = med_chain.manage_langchain(template_text=template_v2, prompt_version="v2")
    summaries_metrics.append({
        "prompt_version": "v2",
        "summary": summary_v2,
        "flesch_score": flesch_reading_ease(summary_v2),
        "word_count": len(summary_v2.split())
    })

    best_summary = med_chain.select_best_summary(summaries_metrics, min_words=20)

    print("\n=== Beste Prompt-Version ===")
    print(f"Version: {best_summary['prompt_version']}")
    print(f"Flesch-Score: {best_summary['flesch_score']}")
    print(f"Wortanzahl: {best_summary['word_count']}")
    print("Zusammenfassung:\n", best_summary["summary"])

