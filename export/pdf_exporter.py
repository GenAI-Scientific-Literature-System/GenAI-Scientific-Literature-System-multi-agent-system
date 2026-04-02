import os
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


def export_report_pdf(result: dict) -> str:
    os.makedirs("exports", exist_ok=True)

    filename = f"exports/final_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

    c = canvas.Canvas(filename, pagesize=A4)
    width, height = A4
    y = height - 40

    def write_line(text, step=16):
        nonlocal y
        c.drawString(40, y, text[:120])
        y -= step
        if y < 60:
            c.showPage()
            y = height - 40

    write_line("Scientific Literature Multi-Agent Analysis Report", 24)
    write_line(f"Query: {result.get('query', '')}", 20)

    write_line("Top Ranked Insights:", 20)
    for idx, item in enumerate(result.get("ranked_insights", [])[:10], start=1):
        if isinstance(item, dict):
            insight = item.get("insight", item.get("claim", ""))
            score = item.get("score", "")
            write_line(f"{idx}. {insight} | score={score}")
        else:
            write_line(f"{idx}. {str(item)}")

    write_line("Execution Summary:", 20)
    execution = result.get("execution", {})
    for agent_name, stats in execution.items():
        write_line(
            f"{agent_name}: status={stats.get('status')} duration={stats.get('duration_sec')}s"
        )

    write_line("Errors:", 20)
    for err in result.get("errors", []):
        write_line(f"{err.get('agent')}: {err.get('error')}")

    c.save()
    return filename