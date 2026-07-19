#!/usr/bin/env python3
"""Build Chapter 2 of Learning STARK using the established book design."""

from __future__ import annotations

import sys
from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Flowable, Frame, NextPageTemplate, PageBreak, Paragraph, Spacer

import build_chapter1_pdf as base


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "book/manuscript/ch02-guided-tour.md"
OUTPUT = ROOT / "output/pdf/learning-stark-chapter-02.pdf"


class ProgramAnatomyDiagram(base.Diagram):
    caption = "Figure 2-1. Top-level items establish the types and functions coordinated by main."

    def __init__(self):
        super().__init__(154)

    def draw(self):
        c = self.canv
        c.saveState()
        y_top = 92
        items = [
            ("STRUCT", "ImageSize", 0, base.BLUE_PALE, base.BLUE),
            ("ENUM", "SizeError", 103, HexColor("#F2E6D7"), base.RUST),
            ("IMPL", "ImageSize", 206, HexColor("#E7ECE8"), base.GREEN),
            ("FUNCTION", "validate_size", 309, base.PAPER_DARK, base.GOLD),
        ]
        for label, name, x, fill, accent in items:
            c.setFillColor(fill)
            c.setStrokeColor(base.GRID)
            c.roundRect(x, y_top, 91, 39, 5, fill=1, stroke=1)
            c.setFillColor(accent)
            c.rect(x, y_top, 4, 39, fill=1, stroke=0)
            c.setFillColor(base.MUTED)
            c.setFont("Arial-Bold", 5.8)
            c.drawString(x + 11, y_top + 25, label)
            c.setFillColor(base.INK)
            c.setFont("Georgia-Bold", 7.1)
            c.drawString(x + 11, y_top + 11, name)
            c.setStrokeColor(base.GRID)
            c.line(x + 45.5, y_top - 13, base.CONTENT_W / 2, 61)
        base.Diagram._box(self, c, (base.CONTENT_W - 132) / 2, 37, 132, 44, "fn main()<br/><font size='6'>construct, call, handle</font>", fill=base.INK, stroke=base.INK, font=7.2)
        c.setFillColor(base.MUTED)
        c.setFont("Arial", 6.3)
        c.drawCentredString(base.CONTENT_W / 2, 21, "items define contracts; main coordinates behavior")
        c.restoreState()


class BlockValueDiagram(base.Diagram):
    caption = "Figure 2-2. The final semicolon determines whether a block returns a value or Unit."

    def __init__(self):
        super().__init__(146)

    def draw(self):
        c = self.canv
        c.saveState()
        gap = 16
        w = (base.CONTENT_W - gap) / 2
        panels = [
            (0, base.BLUE_PALE, base.BLUE, "VALUE", "width * height", "UInt64"),
            (w + gap, HexColor("#F2E6D7"), base.RUST, "STATEMENT", "width * height;", "Unit"),
        ]
        for x, fill, accent, label, code, result in panels:
            c.setFillColor(fill)
            c.setStrokeColor(base.GRID)
            c.roundRect(x, 30, w, 91, 6, fill=1, stroke=1)
            c.setFillColor(accent)
            c.rect(x, 30, 5, 91, fill=1, stroke=0)
            c.setFont("Arial-Bold", 6.2)
            c.drawString(x + 16, 103, label)
            c.setFillColor(base.CODE_BG)
            c.roundRect(x + 16, 62, w - 32, 27, 4, fill=1, stroke=0)
            c.setFillColor(base.CODE_INK)
            c.setFont("CourierNew", 7.4)
            c.drawCentredString(x + w / 2, 72, code)
            c.setFillColor(base.MUTED)
            c.setFont("Georgia", 7)
            c.drawCentredString(x + w / 2, 44, f"block type: {result}")
        c.restoreState()


class BorrowCallDiagram(base.Diagram):
    caption = "Figure 2-3. <font face='CourierNew'>&amp;size</font> lends access for the call while ownership remains in main."

    def __init__(self):
        super().__init__(130)

    def draw(self):
        c = self.canv
        c.saveState()
        y = 45
        base.Diagram._box(self, c, 0, y, 112, 48, "MAIN<br/><font size='6'>owns size</font>", fill=base.PAPER_DARK)
        base.Diagram._arrow(self, c, 118, y + 31, 168, y + 31, base.BLUE)
        c.setFillColor(base.BLUE)
        c.setFont("CourierNew-Bold", 7)
        c.drawCentredString(143, y + 39, "&size")
        base.Diagram._box(self, c, 174, y, 124, 48, "VALIDATE_SIZE<br/><font size='6'>shared borrow</font>", fill=base.BLUE_PALE, stroke=base.BLUE)
        base.Diagram._arrow(self, c, 304, y + 31, 354, y + 31, base.GREEN)
        c.setFillColor(base.GREEN)
        c.setFont("Arial-Bold", 6.2)
        c.drawCentredString(329, y + 39, "RETURN")
        base.Diagram._box(self, c, 360, y, base.CONTENT_W - 360, 48, "MAIN<br/><font size='6'>still owns size</font>", fill=HexColor("#E7ECE8"), stroke=base.GREEN)
        c.setFillColor(base.MUTED)
        c.setFont("Arial", 6.4)
        c.drawCentredString(base.CONTENT_W / 2, 27, "the reference is temporary; the ownership relationship is static")
        c.restoreState()


class DiagnosticDiagram(base.Diagram):
    caption = "Figure 2-4. A useful diagnostic connects the failed expression to the contract that required something else."

    def __init__(self):
        super().__init__(168)

    def draw(self):
        c = self.canv
        c.saveState()
        c.setFillColor(base.CODE_BG)
        c.roundRect(0, 24, base.CONTENT_W, 126, 6, fill=1, stroke=0)
        c.setFillColor(base.RUST)
        c.rect(0, 24, 5, 126, fill=1, stroke=0)
        c.setFont("Arial-Bold", 7.5)
        c.drawString(15, 132, "error[E0001]: type mismatch")
        c.setFont("CourierNew", 6.7)
        c.setFillColor(HexColor("#B9C5C6"))
        c.drawString(15, 113, "--> src/main.stark:12:9")
        c.setFillColor(base.CODE_INK)
        c.drawString(15, 92, "12 | let max_pixels: UInt32 = 4_000_000u64;")
        c.setFillColor(HexColor("#D8A38F"))
        c.drawString(15, 78, "   |                 ------   ^^^^^^^^^^^^^")
        c.setFont("Arial", 6.4)
        c.drawString(58, 63, "required here")
        c.drawString(225, 63, "found UInt64")
        c.setFillColor(HexColor("#9FC2CA"))
        c.setFont("Arial-Bold", 6.4)
        c.drawString(15, 42, "help: align the annotation and literal type")
        c.restoreState()


class FrontEndDiagram(base.Diagram):
    caption = "Figure 2-5. Each front-end stage adds meaning while preserving source provenance."

    def __init__(self):
        super().__init__(122)

    def draw(self):
        c = self.canv
        c.saveState()
        labels = ["SOURCE", "TOKENS", "AST", "NAMES", "TYPES", "BORROWS"]
        fills = [base.PAPER_DARK, base.PAPER_DARK, base.BLUE_PALE, base.BLUE_PALE, HexColor("#E7ECE8"), HexColor("#F2E6D7")]
        gap = 7
        w = (base.CONTENT_W - gap * 5) / 6
        y = 46
        for i, (label, fill) in enumerate(zip(labels, fills)):
            x = i * (w + gap)
            base.Diagram._box(self, c, x, y, w, 35, label, fill=fill, font=5.9)
            if i < len(labels) - 1:
                base.Diagram._arrow(self, c, x + w + 1, y + 17.5, x + w + gap - 1, y + 17.5)
        c.setFillColor(base.MUTED)
        c.setFont("Arial", 6.3)
        c.drawCentredString(base.CONTENT_W / 2, 29, "lex -> parse -> resolve -> check -> analyze")
        c.restoreState()


CHAPTER_DIAGRAMS = {
    "PROGRAM_ANATOMY": ProgramAnatomyDiagram,
    "BLOCK_VALUE": BlockValueDiagram,
    "BORROW_CALL": BorrowCallDiagram,
    "DIAGNOSTIC": DiagnosticDiagram,
    "FRONT_END": FrontEndDiagram,
}


def cover_page(c, doc):
    c.saveState()
    c.setFillColor(base.CODE_BG)
    c.rect(0, 0, base.PAGE_W, base.PAGE_H, fill=1, stroke=0)
    base.draw_topographic(c, base.PAGE_W * 0.42, base.PAGE_H * 0.11, base.PAGE_W * 0.68, base.PAGE_H * 0.78)
    c.setFillColor(base.RUST)
    c.rect(0, base.PAGE_H - 14, base.PAGE_W, 14, fill=1, stroke=0)
    c.setFont("Arial-Bold", 8)
    c.setFillColor(HexColor("#D9A38F"))
    c.drawString(base.MARGIN_X, base.PAGE_H - 55, "EARLY ACCESS / CORE V1 SPECIFICATION EDITION")
    c.setFont("Georgia-Bold", 37)
    c.setFillColor(base.WHITE)
    c.drawString(base.MARGIN_X, base.PAGE_H - 132, "Learning")
    c.setFillColor(base.RUST)
    c.drawString(base.MARGIN_X, base.PAGE_H - 173, "STARK")
    c.setStrokeColor(HexColor("#5A6466"))
    c.setLineWidth(0.7)
    c.line(base.MARGIN_X, base.PAGE_H - 198, base.PAGE_W - base.MARGIN_X, base.PAGE_H - 198)
    c.setFont("Georgia-Italic", 13.2)
    c.setFillColor(HexColor("#DDD7CB"))
    c.drawString(base.MARGIN_X, base.PAGE_H - 224, "Safe Systems Programming for Typed ML Deployment")
    c.setFont("Arial-Bold", 8.5)
    c.setFillColor(base.WHITE)
    c.drawString(base.MARGIN_X, 84, "CHAPTER 02")
    c.setFont("Georgia-Bold", 16)
    c.drawString(base.MARGIN_X, 59, "A Guided Tour of a STARK Program")
    c.setFont("Arial", 7.2)
    c.setFillColor(HexColor("#AEB7B5"))
    c.drawRightString(base.PAGE_W - base.MARGIN_X, 32, "STARK LANGUAGE PROJECT / JULY 2026")
    c.restoreState()


def body_page(c, doc):
    c.saveState()
    c.setFillColor(base.PAPER)
    c.rect(0, 0, base.PAGE_W, base.PAGE_H, fill=1, stroke=0)
    physical = c.getPageNumber()
    logical = physical - 2
    if logical > 1:
        c.setStrokeColor(base.GRID)
        c.setLineWidth(0.45)
        c.line(base.MARGIN_X, base.PAGE_H - 31, base.PAGE_W - base.MARGIN_X, base.PAGE_H - 31)
        c.setFillColor(base.MUTED)
        c.setFont("Arial-Bold", 6.5)
        if logical % 2 == 0:
            c.drawString(base.MARGIN_X, base.PAGE_H - 23, "LEARNING STARK")
        else:
            c.drawRightString(base.PAGE_W - base.MARGIN_X, base.PAGE_H - 23, "A GUIDED TOUR OF A STARK PROGRAM")
    c.setFillColor(base.MUTED)
    c.setFont("Arial", 7)
    c.drawCentredString(base.PAGE_W / 2, 25, str(logical))
    c.restoreState()


def build() -> None:
    base.register_fonts()
    sty = base.styles()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    base.DIAGRAMS.update(CHAPTER_DIAGRAMS)
    body = base.parse_markdown(SOURCE.read_text(encoding="utf-8"), sty)

    frame = Frame(
        base.MARGIN_X,
        base.BOTTOM,
        base.CONTENT_W,
        base.PAGE_H - base.TOP - base.BOTTOM,
        id="main",
        leftPadding=0,
        rightPadding=0,
        topPadding=0,
        bottomPadding=0,
    )
    doc = base.ChapterDocTemplate(
        str(OUTPUT),
        pagesize=(base.PAGE_W, base.PAGE_H),
        leftMargin=base.MARGIN_X,
        rightMargin=base.MARGIN_X,
        topMargin=base.TOP,
        bottomMargin=base.BOTTOM,
        title="Learning STARK - Chapter 2: A Guided Tour of a STARK Program",
        author="STARK Language Project",
        subject="A guided tour of Core v1 program structure, types, borrowing, results, and diagnostics",
        keywords="STARK, programming language, Core v1, syntax, types, ownership, Result, pattern matching",
    )
    doc.addPageTemplates(
        [
            base.PageTemplate(id="cover", frames=[frame], onPage=cover_page),
            base.PageTemplate(id="front", frames=[frame], onPage=base.front_page),
            base.PageTemplate(id="body", frames=[frame], onPage=body_page),
        ]
    )

    legal = [
        Spacer(1, 250),
        Paragraph("<b>Learning STARK</b>", sty["legal"]),
        Paragraph("Chapter 2 preview: <i>A Guided Tour of a STARK Program</i>", sty["legal"]),
        Spacer(1, 7),
        Paragraph("Early Access manuscript - July 2026", sty["legal"]),
        Paragraph(
            "This chapter documents a language specification under active implementation. Core v1 is a normative draft and has not yet been validated by a conforming compiler. Illustrative diagnostics are identified in the text.",
            sty["legal"],
        ),
        Paragraph(
            "The design and text are original to the STARK Language Project. Product and project names mentioned for interoperability remain the property of their respective owners.",
            sty["legal"],
        ),
        Spacer(1, 10),
        Paragraph("Source manuscript: <font face='CourierNew'>book/manuscript/ch02-guided-tour.md</font>", sty["legal"]),
    ]

    deck = "A language begins to feel real when its rules stop being a list and start working together on one page."
    story: list[Flowable] = [
        Spacer(1, base.PAGE_H - base.TOP - base.BOTTOM - 1),
        NextPageTemplate("front"),
        PageBreak(),
        *legal,
        NextPageTemplate("body"),
        PageBreak(),
        base.ChapterOpener("A Guided Tour of a STARK Program", deck, number="02"),
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
