"""
Generate HTML and PDF report from evaluation results.
"""

import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF

def generate_report(metrics_dict, output_pdf="report.pdf", output_html="report.html"):
    df = pd.DataFrame(metrics_dict).T
    df.to_html(output_html)

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Meta-Learning Imbalance Evaluation Report", ln=True)

    for model, metrics in metrics_dict.items():
        pdf.cell(200, 10, txt=f"{model}: F1={metrics['f1']:.4f}, AUC={metrics['auc']:.4f}", ln=True)

    pdf.output(output_pdf)
    print(f"Generated {output_pdf} and {output_html}")

if __name__ == "__main__":
    results = {
        "Meta-Learner": {"f1": 0.81, "auc": 0.91},
        "SMOTE": {"f1": 0.76, "auc": 0.89}
    }
    generate_report(results)
