#!/usr/bin/env python3
"""Build the designed Chapter 1 preview for Learning STARK."""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.fonts import addMapping
from reportlab.lib.pagesizes import portrait
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    KeepTogether,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    XPreformatted,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "book/manuscript/ch01-safer-deployment.md"
OUTPUT = ROOT / "output/pdf/learning-stark-chapter-01.pdf"

PAGE_W = 7 * inch
PAGE_H = 9.25 * inch
MARGIN_X = 0.68 * inch
TOP = 0.68 * inch
BOTTOM = 0.66 * inch
CONTENT_W = PAGE_W - 2 * MARGIN_X

INK = HexColor("#202426")
MUTED = HexColor("#667074")
PAPER = HexColor("#F7F3EA")
PAPER_DARK = HexColor("#EDE5D7")
WHITE = HexColor("#FFFDF8")
RUST = HexColor("#B84E2F")
RUST_DARK = HexColor("#7D2E1C")
BLUE = HexColor("#2C6473")
BLUE_PALE = HexColor("#E6F0F1")
GOLD = HexColor("#C08B38")
GREEN = HexColor("#47705F")
CODE_BG = HexColor("#1F292D")
CODE_INK = HexColor("#F3EEE2")
GRID = HexColor("#D8CDBC")


def register_fonts() -> None:
    supplemental = Path("/System/Library/Fonts/Supplemental")
    fonts = {
        "Georgia": supplemental / "Georgia.ttf",
        "Georgia-Bold": supplemental / "Georgia Bold.ttf",
        "Georgia-Italic": supplemental / "Georgia Italic.ttf",
        "Georgia-BoldItalic": supplemental / "Georgia Bold Italic.ttf",
        "CourierNew": supplemental / "Courier New.ttf",
        "CourierNew-Bold": supplemental / "Courier New Bold.ttf",
        "Arial": supplemental / "Arial.ttf",
        "Arial-Bold": supplemental / "Arial Bold.ttf",
    }
    for name, path in fonts.items():
        if path.exists():
            pdfmetrics.registerFont(TTFont(name, str(path)))
    addMapping("Georgia", 0, 0, "Georgia")
    addMapping("Georgia", 1, 0, "Georgia-Bold")
    addMapping("Georgia", 0, 1, "Georgia-Italic")
    addMapping("Georgia", 1, 1, "Georgia-BoldItalic")
    addMapping("CourierNew", 0, 0, "CourierNew")
    addMapping("CourierNew", 1, 0, "CourierNew-Bold")
    addMapping("Arial", 0, 0, "Arial")
    addMapping("Arial", 1, 0, "Arial-Bold")


def styles() -> dict[str, ParagraphStyle]:
    sample = getSampleStyleSheet()
    return {
        "body": ParagraphStyle(
            "Body",
            parent=sample["BodyText"],
            fontName="Georgia",
            fontSize=9.55,
            leading=14.25,
            textColor=INK,
            alignment=TA_JUSTIFY,
            spaceAfter=7.2,
            allowWidows=0,
            allowOrphans=0,
            splitLongWords=False,
        ),
        "lead": ParagraphStyle(
            "Lead",
            fontName="Georgia",
            fontSize=11.5,
            leading=17.2,
            textColor=INK,
            alignment=TA_LEFT,
            spaceAfter=14,
        ),
        "h2": ParagraphStyle(
            "Heading2",
            fontName="Georgia-Bold",
            fontSize=17.2,
            leading=21,
            textColor=INK,
            spaceBefore=15,
            spaceAfter=8,
            keepWithNext=True,
        ),
        "h3": ParagraphStyle(
            "Heading3",
            fontName="Arial-Bold",
            fontSize=10.2,
            leading=13,
            textColor=RUST_DARK,
            spaceBefore=10,
            spaceAfter=4.5,
            keepWithNext=True,
        ),
        "bullet": ParagraphStyle(
            "Bullet",
            fontName="Georgia",
            fontSize=9.3,
            leading=13.6,
            textColor=INK,
            leftIndent=16,
            firstLineIndent=-10,
            bulletIndent=4,
            spaceAfter=3.5,
        ),
        "number": ParagraphStyle(
            "Number",
            fontName="Georgia",
            fontSize=9.3,
            leading=13.6,
            textColor=INK,
            leftIndent=20,
            firstLineIndent=-16,
            spaceAfter=3.5,
        ),
        "quote": ParagraphStyle(
            "Quote",
            fontName="Georgia-Italic",
            fontSize=12.1,
            leading=17.3,
            textColor=RUST_DARK,
            leftIndent=22,
            rightIndent=16,
            spaceBefore=7,
            spaceAfter=12,
        ),
        "callout": ParagraphStyle(
            "Callout",
            fontName="Georgia",
            fontSize=8.85,
            leading=13.15,
            textColor=INK,
            spaceAfter=0,
        ),
        "caption": ParagraphStyle(
            "Caption",
            fontName="Arial",
            fontSize=7.3,
            leading=9.5,
            textColor=MUTED,
            alignment=TA_CENTER,
            spaceBefore=4,
            spaceAfter=8,
        ),
        "source": ParagraphStyle(
            "SourceNote",
            fontName="Georgia",
            fontSize=8.1,
            leading=12.1,
            textColor=INK,
            leftIndent=16,
            firstLineIndent=-12,
            spaceAfter=4,
        ),
        "legal": ParagraphStyle(
            "Legal",
            fontName="Georgia",
            fontSize=8.1,
            leading=12.2,
            textColor=MUTED,
            alignment=TA_LEFT,
            spaceAfter=7,
        ),
    }


def inline_markup(text: str) -> str:
    parts = text.split("`")
    rendered: list[str] = []
    for index, part in enumerate(parts):
        escaped = html.escape(part, quote=False)
        if index % 2:
            rendered.append(f'<font face="CourierNew" color="#7D2E1C">{escaped}</font>')
        else:
            escaped = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", escaped)
            escaped = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", escaped)
            rendered.append(escaped)
    return "".join(rendered)


class ChapterDocTemplate(BaseDocTemplate):
    def afterFlowable(self, flowable: Flowable) -> None:
        if isinstance(flowable, Paragraph) and flowable.style.name in {"Heading2", "Heading3"}:
            text = flowable.getPlainText()
            key = f"heading-{self.seq.nextf('heading')}"
            level = 0 if flowable.style.name == "Heading2" else 1
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(text, key, level=level, closed=False)


class ChapterOpener(Flowable):
    def __init__(self, title: str, deck: str, number: str = "01"):
        super().__init__()
        self.title = title
        self.deck = deck
        self.number = number
        self.width = CONTENT_W
        self.height = 238

    def draw(self) -> None:
        c = self.canv
        c.saveState()
        c.setFillColor(RUST)
        c.rect(0, self.height - 9, 42, 4, fill=1, stroke=0)
        c.setFont("Arial-Bold", 64)
        c.setFillColor(PAPER_DARK)
        c.drawString(self.width - 94, self.height - 50, self.number)
        c.setFont("Arial-Bold", 8.2)
        c.setFillColor(RUST)
        c.drawString(0, self.height - 34, f"CHAPTER {self.number}")
        title = Paragraph(
            inline_markup(self.title),
            ParagraphStyle(
                "ChapterTitle",
                fontName="Georgia-Bold",
                fontSize=29,
                leading=33.5,
                textColor=INK,
                spaceAfter=0,
            ),
        )
        _, th = title.wrap(self.width - 32, 100)
        title.drawOn(c, 0, self.height - 76 - th)
        c.setStrokeColor(GRID)
        c.setLineWidth(0.7)
        c.line(0, 74, self.width, 74)
        deck = Paragraph(
            inline_markup(self.deck),
            ParagraphStyle(
                "ChapterDeck",
                fontName="Georgia-Italic",
                fontSize=11.7,
                leading=16.8,
                textColor=RUST_DARK,
            ),
        )
        _, dh = deck.wrap(self.width - 34, 62)
        deck.drawOn(c, 18, 53 - dh / 2)
        c.restoreState()


class CodePanel(Flowable):
    def __init__(self, code: str, label: str | None = None):
        super().__init__()
        self.code = code.rstrip()
        self.label = label
        self.pad = 11
        self.label_h = 17 if label else 0
        self.width = CONTENT_W
        self.pre = XPreformatted(
            html.escape(self.code),
            ParagraphStyle(
                "Code",
                fontName="CourierNew",
                fontSize=7.15,
                leading=9.55,
                textColor=CODE_INK,
                leftIndent=0,
                rightIndent=0,
                spaceAfter=0,
            ),
        )

    def wrap(self, avail_width: float, avail_height: float) -> tuple[float, float]:
        self.width = avail_width
        _, ph = self.pre.wrap(avail_width - 2 * self.pad, avail_height)
        self.height = ph + 2 * self.pad + self.label_h
        return avail_width, self.height

    def split(self, avail_width: float, avail_height: float) -> list[Flowable]:
        if self.height <= avail_height:
            return [self]
        return []

    def draw(self) -> None:
        c = self.canv
        c.saveState()
        c.setFillColor(CODE_BG)
        c.roundRect(0, 0, self.width, self.height, 6, fill=1, stroke=0)
        c.setFillColor(RUST)
        c.roundRect(0, 0, 4, self.height, 2, fill=1, stroke=0)
        if self.label:
            c.setFont("Arial-Bold", 6.8)
            c.setFillColor(HexColor("#D8A38F"))
            c.drawString(self.pad, self.height - 13, self.label.upper())
        self.pre.drawOn(c, self.pad, self.pad)
        c.restoreState()


class Diagram(Flowable):
    caption = ""

    def __init__(self, height: float):
        super().__init__()
        self.width = CONTENT_W
        self.height = height

    def _box(self, c, x, y, w, h, label, fill=WHITE, stroke=GRID, font=7.1):
        c.setFillColor(fill)
        c.setStrokeColor(stroke)
        c.setLineWidth(0.7)
        c.roundRect(x, y, w, h, 5, fill=1, stroke=1)
        p = Paragraph(
            label,
            ParagraphStyle(
                "DiagramLabel",
                fontName="Arial-Bold",
                fontSize=font,
                leading=font + 1.5,
                textColor=WHITE if fill == INK else INK,
                alignment=TA_CENTER,
            ),
        )
        _, ph = p.wrap(w - 8, h - 4)
        p.drawOn(c, x + 4, y + (h - ph) / 2)

    def _arrow(self, c, x1, y1, x2, y2, color=MUTED):
        c.setStrokeColor(color)
        c.setFillColor(color)
        c.setLineWidth(1)
        c.line(x1, y1, x2, y2)
        c.line(x2, y2, x2 - 4, y2 + 2.5)
        c.line(x2, y2, x2 - 4, y2 - 2.5)


class PipelineDiagram(Diagram):
    caption = "Figure 1-1. The checked program surrounds, rather than replaces, the model backend."

    def __init__(self):
        super().__init__(112)

    def draw(self):
        c = self.canv
        c.saveState()
        labels = ["Request", "Decode", "Refine", "Transform", "Model", "Response"]
        fills = [PAPER_DARK, PAPER_DARK, BLUE_PALE, BLUE_PALE, HexColor("#F0E5D6"), BLUE_PALE]
        gap = 7
        w = (self.width - gap * 5) / 6
        y = 43
        for i, (label, fill) in enumerate(zip(labels, fills)):
            x = i * (w + gap)
            self._box(c, x, y, w, 34, label, fill=fill)
            if i < len(labels) - 1:
                self._arrow(c, x + w + 1, y + 17, x + w + gap - 1, y + 17)
        c.setFont("Arial-Bold", 6.4)
        c.setFillColor(RUST_DARK)
        c.drawCentredString(w + gap / 2, 26, "DYNAMIC EVIDENCE")
        c.setFillColor(BLUE)
        c.drawCentredString(3.5 * (w + gap) - gap / 2, 26, "STATICALLY CHECKED REGION")
        c.setStrokeColor(GRID)
        c.line(0, 18, 2 * w + gap, 18)
        c.line(2 * (w + gap), 18, self.width, 18)
        c.restoreState()


class FiveConcernsDiagram(Diagram):
    caption = "Figure 1-2. Five questions expose underspecified deployment interfaces."

    def __init__(self):
        super().__init__(150)

    def draw(self):
        c = self.canv
        c.saveState()
        center_w, center_h = 124, 38
        cx = (self.width - center_w) / 2
        cy = 54
        items = [
            ("SHAPE", 0, 100, BLUE_PALE),
            ("MEANING", self.width - 76, 100, HexColor("#F2E6D7")),
            ("LOCATION", 0, 22, HexColor("#E8EEE6")),
            ("OWNERSHIP", self.width - 76, 22, HexColor("#EFE6EC")),
            ("CONSTRAINTS", (self.width - 86) / 2, 4, PAPER_DARK),
        ]
        points = [
            (cx, cy + center_h),
            (cx + center_w, cy + center_h),
            (cx, cy),
            (cx + center_w, cy),
            (cx + center_w / 2, cy),
        ]
        targets = [(76, 116), (self.width - 76, 116), (76, 39), (self.width - 76, 39), (self.width / 2, 38)]
        c.setStrokeColor(GRID)
        c.setLineWidth(0.8)
        for (sx, sy), (tx, ty) in zip(points, targets):
            c.line(sx, sy, tx, ty)
        self._box(c, cx, cy, center_w, center_h, "DEPLOYMENT<br/>CONTRACT", fill=INK, stroke=INK, font=7.3)
        for label, x, y, fill in items:
            w = 86 if label == "CONSTRAINTS" else 76
            self._box(c, x, y, w, 31, label, fill=fill, font=6.7)
        c.restoreState()


class RefinementDiagram(Diagram):
    caption = "Figure 1-3. Runtime evidence is checked once, then preserved as static type information."

    def __init__(self):
        super().__init__(124)

    def draw(self):
        c = self.canv
        c.saveState()
        y = 38
        self._box(c, 0, y, 110, 48, "TensorAny<br/><font size='6'>untrusted dtype and shape</font>", fill=PAPER_DARK)
        self._arrow(c, 114, y + 24, 148, y + 24, RUST)
        self._box(c, 153, y - 2, 88, 52, "REFINE<br/><font size='6'>runtime validation</font>", fill=HexColor("#F2E3D7"), stroke=RUST)
        self._arrow(c, 246, y + 24, 280, y + 24, BLUE)
        self._box(c, 285, y, self.width - 285, 48, "Tensor&lt;UInt8,<br/>[B, 224, 224, 3]&gt;", fill=BLUE_PALE, stroke=BLUE, font=6.7)
        c.setFont("Arial", 6.5)
        c.setFillColor(MUTED)
        c.drawString(1, 24, "may fail here")
        c.drawRightString(self.width - 1, 24, "shape and dtype are checked from here onward")
        c.restoreState()


class LayersDiagram(Diagram):
    caption = "Figure 1-4. Core, extension, and backend have separate responsibilities."

    def __init__(self):
        super().__init__(158)

    def draw(self):
        c = self.canv
        c.saveState()
        rows = [
            ("TENSOR EXTENSION", "shapes, dtypes, devices, model signatures", BLUE_PALE, BLUE),
            ("CORE V1", "types, ownership, errors, modules, collections", PAPER_DARK, RUST_DARK),
            ("EXISTING BACKEND", "ONNX Runtime, IREE, generated Rust/C", HexColor("#E7ECE8"), GREEN),
        ]
        y = 104
        for title, desc, fill, accent in rows:
            c.setFillColor(fill)
            c.setStrokeColor(GRID)
            c.roundRect(18, y, self.width - 36, 38, 5, fill=1, stroke=1)
            c.setFillColor(accent)
            c.rect(18, y, 5, 38, fill=1, stroke=0)
            c.setFillColor(INK)
            c.setFont("Arial-Bold", 7.2)
            c.drawString(34, y + 22, title)
            c.setFont("Georgia", 7.1)
            c.setFillColor(MUTED)
            c.drawString(34, y + 10, desc)
            if y > 28:
                c.setStrokeColor(GRID)
                c.line(self.width / 2, y - 8, self.width / 2, y)
            y -= 48
        c.restoreState()


DIAGRAMS = {
    "PIPELINE": PipelineDiagram,
    "FIVE_CONCERNS": FiveConcernsDiagram,
    "REFINEMENT": RefinementDiagram,
    "LAYERS": LayersDiagram,
}


def callout(kind: str, content: str, sty: dict[str, ParagraphStyle]) -> Table:
    palette = {
        "status": (HexColor("#F0E4D9"), RUST),
        "principle": (BLUE_PALE, BLUE),
        "note": (HexColor("#ECEEE7"), GREEN),
    }
    bg, accent = palette.get(kind, (PAPER_DARK, RUST))
    p = Paragraph(inline_markup(content).replace("\n\n", "<br/><br/>"), sty["callout"])
    box = Table([[Spacer(1, 1), p]], colWidths=[5, CONTENT_W - 5 - 20], hAlign="LEFT")
    box.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), bg),
                ("BACKGROUND", (0, 0), (0, 0), accent),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (0, 0), 0),
                ("RIGHTPADDING", (0, 0), (0, 0), 0),
                ("TOPPADDING", (0, 0), (0, 0), 0),
                ("BOTTOMPADDING", (0, 0), (0, 0), 0),
                ("LEFTPADDING", (1, 0), (1, 0), 11),
                ("RIGHTPADDING", (1, 0), (1, 0), 11),
                ("TOPPADDING", (1, 0), (1, 0), 9),
                ("BOTTOMPADDING", (1, 0), (1, 0), 9),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.Color(accent.red, accent.green, accent.blue, alpha=0.25)),
            ]
        )
    )
    return box


def diagram_with_caption(name: str, sty: dict[str, ParagraphStyle]) -> list[Flowable]:
    diagram = DIAGRAMS[name]()
    return [Spacer(1, 6), diagram, Paragraph(diagram.caption, sty["caption"])]


def parse_markdown(text: str, sty: dict[str, ParagraphStyle]) -> list[Flowable]:
    lines = text.splitlines()
    story: list[Flowable] = []
    paragraph: list[str] = []
    in_code = False
    code_lines: list[str] = []
    code_label: str | None = None
    in_callout = False
    callout_kind = "note"
    callout_lines: list[str] = []
    skipped_title = False
    skipped_opening_quote = False
    source_mode = False
    body_paragraph_count = 0

    def flush_paragraph() -> None:
        nonlocal paragraph, body_paragraph_count
        if paragraph:
            joined = " ".join(x.strip() for x in paragraph)
            style = sty["lead"] if body_paragraph_count == 0 else sty["body"]
            story.append(Paragraph(inline_markup(joined), style))
            body_paragraph_count += 1
            paragraph = []

    for line in lines:
        if in_code:
            if line.startswith("```"):
                story.extend([Spacer(1, 5), CodePanel("\n".join(code_lines), code_label), Spacer(1, 7)])
                in_code = False
                code_lines = []
                code_label = None
            else:
                code_lines.append(line)
            continue

        if in_callout:
            if line.strip() == ":::":
                flush_paragraph()
                story.extend([Spacer(1, 6), callout(callout_kind, "\n".join(callout_lines), sty), Spacer(1, 9)])
                in_callout = False
                callout_lines = []
            else:
                callout_lines.append(line)
            continue

        if line.startswith(":::"):
            flush_paragraph()
            in_callout = True
            callout_kind = line[3:].strip()
            continue

        if line.startswith("```"):
            flush_paragraph()
            in_code = True
            code_label = line[3:].strip() or None
            continue

        marker = re.fullmatch(r"\[\[DIAGRAM:([A-Z_]+)\]\]", line.strip())
        if marker:
            flush_paragraph()
            story.extend(diagram_with_caption(marker.group(1), sty))
            continue

        if line.strip() == "[[PAGEBREAK]]":
            flush_paragraph()
            story.append(PageBreak())
            continue

        if line.startswith("# ") and not skipped_title:
            skipped_title = True
            continue

        if line.startswith("> ") and not skipped_opening_quote:
            skipped_opening_quote = True
            continue

        if line.startswith("## "):
            flush_paragraph()
            text_value = line[3:].strip()
            source_mode = text_value == "Source notes"
            story.append(Paragraph(inline_markup(text_value), sty["h2"]))
            continue

        if line.startswith("### "):
            flush_paragraph()
            story.append(Paragraph(inline_markup(line[4:].strip()), sty["h3"]))
            continue

        if line.startswith("> "):
            flush_paragraph()
            story.append(Paragraph(inline_markup(line[2:].strip()), sty["quote"]))
            continue

        bullet = re.match(r"^- (.+)$", line)
        if bullet:
            flush_paragraph()
            story.append(Paragraph("&#8226;&nbsp;&nbsp;" + inline_markup(bullet.group(1)), sty["bullet"]))
            continue

        number = re.match(r"^(\d+)\. (.+)$", line)
        if number:
            flush_paragraph()
            selected = sty["source"] if source_mode else sty["number"]
            story.append(Paragraph(f"<b>{number.group(1)}.</b>&nbsp;&nbsp;" + inline_markup(number.group(2)), selected))
            continue

        if not line.strip():
            flush_paragraph()
        else:
            paragraph.append(line)

    flush_paragraph()
    return story


def draw_topographic(c, x0, y0, w, h):
    c.saveState()
    c.setStrokeColor(HexColor("#394347"))
    c.setLineWidth(0.35)
    for i in range(12):
        inset = i * 6.5
        c.roundRect(x0 + inset, y0 + inset * 0.42, w - 2 * inset, h - inset * 0.84, 18 + i * 2, fill=0, stroke=1)
    c.setStrokeColor(RUST)
    c.setLineWidth(0.8)
    c.line(x0 + w * 0.16, y0 + h * 0.5, x0 + w * 0.84, y0 + h * 0.5)
    c.line(x0 + w * 0.5, y0 + h * 0.15, x0 + w * 0.5, y0 + h * 0.85)
    c.restoreState()


def cover_page(c, doc):
    c.saveState()
    c.setFillColor(CODE_BG)
    c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    draw_topographic(c, PAGE_W * 0.42, PAGE_H * 0.11, PAGE_W * 0.68, PAGE_H * 0.78)
    c.setFillColor(RUST)
    c.rect(0, PAGE_H - 14, PAGE_W, 14, fill=1, stroke=0)
    c.setFont("Arial-Bold", 8)
    c.setFillColor(HexColor("#D9A38F"))
    c.drawString(MARGIN_X, PAGE_H - 55, "EARLY ACCESS / CORE V1 SPECIFICATION EDITION")
    c.setFont("Georgia-Bold", 37)
    c.setFillColor(WHITE)
    c.drawString(MARGIN_X, PAGE_H - 132, "Learning")
    c.setFillColor(RUST)
    c.drawString(MARGIN_X, PAGE_H - 173, "STARK")
    c.setStrokeColor(HexColor("#5A6466"))
    c.setLineWidth(0.7)
    c.line(MARGIN_X, PAGE_H - 198, PAGE_W - MARGIN_X, PAGE_H - 198)
    c.setFont("Georgia-Italic", 13.2)
    c.setFillColor(HexColor("#DDD7CB"))
    c.drawString(MARGIN_X, PAGE_H - 224, "Safe Systems Programming for Typed ML Deployment")
    c.setFont("Arial-Bold", 8.5)
    c.setFillColor(WHITE)
    c.drawString(MARGIN_X, 84, "CHAPTER 01")
    c.setFont("Georgia-Bold", 16)
    c.drawString(MARGIN_X, 59, "A Language for Safer Deployment")
    c.setFont("Arial", 7.2)
    c.setFillColor(HexColor("#AEB7B5"))
    c.drawRightString(PAGE_W - MARGIN_X, 32, "STARK LANGUAGE PROJECT / JULY 2026")
    c.restoreState()


def front_page(c, doc):
    c.saveState()
    c.setFillColor(PAPER)
    c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    c.restoreState()


def body_page(c, doc):
    c.saveState()
    c.setFillColor(PAPER)
    c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    physical = c.getPageNumber()
    logical = physical - 2
    if logical > 1:
        c.setStrokeColor(GRID)
        c.setLineWidth(0.45)
        c.line(MARGIN_X, PAGE_H - 31, PAGE_W - MARGIN_X, PAGE_H - 31)
        c.setFillColor(MUTED)
        c.setFont("Arial-Bold", 6.5)
        if logical % 2 == 0:
            c.drawString(MARGIN_X, PAGE_H - 23, "LEARNING STARK")
        else:
            c.drawRightString(PAGE_W - MARGIN_X, PAGE_H - 23, "A LANGUAGE FOR SAFER DEPLOYMENT")
    c.setFillColor(MUTED)
    c.setFont("Arial", 7)
    c.drawCentredString(PAGE_W / 2, 25, str(logical))
    c.restoreState()


def build() -> None:
    register_fonts()
    sty = styles()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    text = SOURCE.read_text(encoding="utf-8")
    body = parse_markdown(text, sty)

    frame = Frame(MARGIN_X, BOTTOM, CONTENT_W, PAGE_H - TOP - BOTTOM, id="main", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    doc = ChapterDocTemplate(
        str(OUTPUT),
        pagesize=portrait((PAGE_W, PAGE_H)),
        leftMargin=MARGIN_X,
        rightMargin=MARGIN_X,
        topMargin=TOP,
        bottomMargin=BOTTOM,
        title="Learning STARK - Chapter 1: A Language for Safer Deployment",
        author="STARK Language Project",
        subject="An introduction to STARK Core v1 and the tensor deployment extension",
        keywords="STARK, programming language, ownership, tensors, ONNX, machine learning deployment",
    )
    doc.addPageTemplates(
        [
            PageTemplate(id="cover", frames=[frame], onPage=cover_page),
            PageTemplate(id="front", frames=[frame], onPage=front_page),
            PageTemplate(id="body", frames=[frame], onPage=body_page),
        ]
    )

    legal = [
        Spacer(1, 250),
        Paragraph("<b>Learning STARK</b>", sty["legal"]),
        Paragraph("Chapter 1 preview: <i>A Language for Safer Deployment</i>", sty["legal"]),
        Spacer(1, 7),
        Paragraph("Early Access manuscript - July 2026", sty["legal"]),
        Paragraph(
            "This chapter documents a language specification under active implementation. Core v1 and tensor extension v0.1 are normative drafts and have not yet been validated by a conforming compiler. Illustrative commands and diagnostics are identified in the text.",
            sty["legal"],
        ),
        Paragraph(
            "The design and text are original to the STARK Language Project. Product and project names mentioned for interoperability remain the property of their respective owners.",
            sty["legal"],
        ),
        Spacer(1, 10),
        Paragraph("Source manuscript: <font face='CourierNew'>book/manuscript/ch01-safer-deployment.md</font>", sty["legal"]),
    ]

    deck = "The most expensive deployment defect is often not a sophisticated numerical failure. It is an ordinary disagreement about what a value is."
    story: list[Flowable] = [
        Spacer(1, PAGE_H - TOP - BOTTOM - 1),
        NextPageTemplate("front"),
        PageBreak(),
        *legal,
        NextPageTemplate("body"),
        PageBreak(),
        ChapterOpener("A Language for Safer Deployment", deck),
        *body,
    ]
    doc.build(story)
    print(OUTPUT)


if __name__ == "__main__":
    try:
        build()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise
