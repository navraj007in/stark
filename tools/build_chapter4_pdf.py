#!/usr/bin/env python3
"""Build Chapter 4 of Learning STARK using the established book design."""

from __future__ import annotations

import sys
from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.platypus import Flowable, Frame, NextPageTemplate, PageBreak, Paragraph, Spacer

import build_chapter1_pdf as base


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "book/manuscript/ch04-control-flow-pattern-matching.md"
OUTPUT = ROOT / "output/pdf/learning-stark-chapter-04.pdf"


class FlowAsValueDiagram(base.Diagram):
    caption = "Figure 4-1. Typed control flow selects one path while preserving one result contract."

    def __init__(self):
        super().__init__(158)

    def draw(self):
        c = self.canv
        c.saveState()
        base.Diagram._box(self, c, 150, 102, 100, 34, "BOOL<br/><font size='6'>condition</font>", fill=base.INK, stroke=base.INK, font=7)
        base.Diagram._arrow(self, c, 178, 100, 91, 73, base.BLUE)
        base.Diagram._arrow(self, c, 222, 100, 309, 73, base.RUST)
        c.setFont("Arial-Bold", 5.8)
        c.setFillColor(base.BLUE)
        c.drawString(119, 89, "TRUE")
        c.setFillColor(base.RUST)
        c.drawString(263, 89, "FALSE")
        base.Diagram._box(self, c, 20, 35, 142, 38, "BRANCH A<br/><font size='6'>value: T</font>", fill=base.BLUE_PALE, stroke=base.BLUE, font=7)
        base.Diagram._box(self, c, 238, 35, 142, 38, "BRANCH B<br/><font size='6'>value: T</font>", fill=HexColor("#F2E6D7"), stroke=base.RUST, font=7)
        base.Diagram._arrow(self, c, 163, 54, 195, 24, base.GRID)
        base.Diagram._arrow(self, c, 237, 54, 205, 24, base.GRID)
        c.setFillColor(base.INK)
        c.setFont("Georgia-Bold", 7)
        c.drawCentredString(200, 12, "whole expression: T")
        c.restoreState()


class LoopValueDiagram(base.Diagram):
    caption = "Figure 4-2. A plain loop can return a value through a typed break edge."

    def __init__(self):
        super().__init__(146)

    def draw(self):
        c = self.canv
        c.saveState()
        c.setFillColor(base.BLUE_PALE)
        c.setStrokeColor(base.BLUE)
        c.roundRect(25, 27, 190, 94, 8, fill=1, stroke=1)
        c.setFillColor(base.INK)
        c.setFont("Georgia-Bold", 9)
        c.drawCentredString(120, 97, "loop body")
        c.setFillColor(base.MUTED)
        c.setFont("Arial", 6.5)
        c.drawCentredString(120, 76, "repeat while no exit is selected")
        base.Diagram._arrow(self, c, 89, 52, 151, 52, base.BLUE)
        c.setFillColor(base.BLUE)
        c.setFont("CourierNew-Bold", 7)
        c.drawCentredString(120, 38, "next iteration")
        base.Diagram._arrow(self, c, 216, 75, 275, 75, base.RUST)
        c.setFillColor(base.RUST)
        c.setFont("CourierNew-Bold", 7)
        c.drawCentredString(245, 86, "break value;")
        base.Diagram._box(self, c, 282, 52, 98, 46, "RESULT<br/><font size='6'>type T</font>", fill=HexColor("#F2E6D7"), stroke=base.RUST, font=7)
        c.restoreState()


class ForProtocolDiagram(base.Diagram):
    caption = "Figure 4-3. A for loop receives successive Item values from an Iterator."

    def __init__(self):
        super().__init__(136)

    def draw(self):
        c = self.canv
        c.saveState()
        boxes = [(0, "ITERABLE", "range / iter()", base.PAPER_DARK, base.GRID), (145, "ITERATOR", "next()", base.BLUE_PALE, base.BLUE), (290, "LOOP BINDING", "Item", HexColor("#E7ECE8"), base.GREEN)]
        for x, title, detail, fill, stroke in boxes:
            base.Diagram._box(self, c, x, 50, 110, 48, f"{title}<br/><font size='6'>{detail}</font>", fill=fill, stroke=stroke, font=6.5)
        base.Diagram._arrow(self, c, 115, 74, 139, 74, base.BLUE)
        base.Diagram._arrow(self, c, 260, 74, 284, 74, base.GREEN)
        c.setFillColor(base.MUTED)
        c.setFont("Arial", 6.4)
        c.drawCentredString(200, 30, "the iterator implementation defines both sequence and ownership behavior")
        c.restoreState()


class ExhaustivenessDiagram(base.Diagram):
    caption = "Figure 4-4. Exhaustiveness requires every variant to reach an arm or a catch-all."

    def __init__(self):
        super().__init__(158)

    def draw(self):
        c = self.canv
        c.saveState()
        colors = [("Healthy", base.BLUE_PALE, base.BLUE), ("Degraded", HexColor("#E7ECE8"), base.GREEN), ("Unavailable", HexColor("#F2E6D7"), base.RUST)]
        for i, (name, fill, stroke) in enumerate(colors):
            y = 103 - i * 37
            base.Diagram._box(self, c, 0, y, 112, 27, name, fill=fill, stroke=stroke, font=6.5)
            base.Diagram._arrow(self, c, 116, y + 13, 190, y + 13, stroke)
        c.setFillColor(base.PAPER_DARK)
        c.setStrokeColor(base.GRID)
        c.roundRect(198, 28, 202, 105, 7, fill=1, stroke=1)
        c.setFillColor(base.INK)
        c.setFont("Georgia-Bold", 8)
        c.drawString(214, 111, "match arms")
        c.setFont("CourierNew", 6.4)
        c.drawString(214, 88, "Healthy     => true")
        c.drawString(214, 67, "Degraded    => true")
        c.setFillColor(base.RUST)
        c.drawString(214, 46, "Unavailable => false")
        c.restoreState()


class OptionFlowDiagram(base.Diagram):
    caption = "Figure 4-5. Option makes both presence and absence explicit in the value and the control flow."

    def __init__(self):
        super().__init__(145)

    def draw(self):
        c = self.canv
        c.saveState()
        base.Diagram._box(self, c, 0, 54, 105, 42, "Option&lt;T&gt;", fill=base.INK, stroke=base.INK, font=7)
        base.Diagram._arrow(self, c, 109, 75, 162, 103, base.BLUE)
        base.Diagram._arrow(self, c, 109, 75, 162, 47, base.RUST)
        base.Diagram._box(self, c, 169, 88, 103, 35, "Some(value)<br/><font size='6'>bind T</font>", fill=base.BLUE_PALE, stroke=base.BLUE, font=6.8)
        base.Diagram._box(self, c, 169, 28, 103, 35, "None<br/><font size='6'>no payload</font>", fill=HexColor("#F2E6D7"), stroke=base.RUST, font=6.8)
        base.Diagram._arrow(self, c, 278, 105, 326, 105, base.GREEN)
        base.Diagram._arrow(self, c, 278, 45, 326, 45, base.GREEN)
        base.Diagram._box(self, c, 333, 54, 67, 42, "RESULT<br/><font size='6'>one type</font>", fill=HexColor("#E7ECE8"), stroke=base.GREEN, font=6.4)
        c.restoreState()


class StateTransitionDiagram(base.Diagram):
    caption = "Figure 4-6. Events choose typed transitions; the fallback policy preserves the current phase."

    def __init__(self):
        super().__init__(154)

    def draw(self):
        c = self.canv
        c.saveState()
        nodes = [(0, 82, "IDLE", base.PAPER_DARK, base.GRID), (145, 82, "LOADING", base.BLUE_PALE, base.BLUE), (290, 82, "SERVING", HexColor("#E7ECE8"), base.GREEN), (145, 20, "FAILED", HexColor("#F2E6D7"), base.RUST)]
        for x, y, name, fill, stroke in nodes:
            base.Diagram._box(self, c, x, y, 110, 34, name, fill=fill, stroke=stroke, font=6.8)
        base.Diagram._arrow(self, c, 115, 99, 139, 99, base.BLUE)
        base.Diagram._arrow(self, c, 260, 99, 284, 99, base.GREEN)
        base.Diagram._arrow(self, c, 200, 79, 200, 58, base.RUST)
        base.Diagram._arrow(self, c, 345, 80, 260, 45, base.RUST)
        c.setFont("Arial-Bold", 5.5)
        c.setFillColor(base.BLUE)
        c.drawCentredString(127, 110, "LOAD")
        c.setFillColor(base.GREEN)
        c.drawCentredString(272, 110, "LOADED")
        c.setFillColor(base.RUST)
        c.drawString(208, 66, "REJECTED")
        c.drawString(304, 59, "FAILURE")
        c.restoreState()


CHAPTER_DIAGRAMS = {
    "FLOW_AS_VALUE": FlowAsValueDiagram,
    "LOOP_VALUE": LoopValueDiagram,
    "FOR_PROTOCOL": ForProtocolDiagram,
    "EXHAUSTIVENESS": ExhaustivenessDiagram,
    "OPTION_FLOW": OptionFlowDiagram,
    "STATE_TRANSITION": StateTransitionDiagram,
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
    c.line(base.MARGIN_X, base.PAGE_H - 198, base.PAGE_W - base.MARGIN_X, base.PAGE_H - 198)
    c.setFont("Georgia-Italic", 13.2)
    c.setFillColor(HexColor("#DDD7CB"))
    c.drawString(base.MARGIN_X, base.PAGE_H - 224, "Safe Systems Programming for Typed ML Deployment")
    c.setFont("Arial-Bold", 8.5)
    c.setFillColor(base.WHITE)
    c.drawString(base.MARGIN_X, 84, "CHAPTER 04")
    c.setFont("Georgia-Bold", 16)
    c.drawString(base.MARGIN_X, 59, "Control Flow and Pattern Matching")
    c.setFont("Arial", 7.2)
    c.setFillColor(HexColor("#AEB7B5"))
    c.drawRightString(base.PAGE_W - base.MARGIN_X, 32, "STARK LANGUAGE PROJECT / JULY 2026")
    c.restoreState()


def body_page(c, doc):
    c.saveState()
    c.setFillColor(base.PAPER)
    c.rect(0, 0, base.PAGE_W, base.PAGE_H, fill=1, stroke=0)
    logical = c.getPageNumber() - 2
    if logical > 1:
        c.setStrokeColor(base.GRID)
        c.setLineWidth(0.45)
        c.line(base.MARGIN_X, base.PAGE_H - 31, base.PAGE_W - base.MARGIN_X, base.PAGE_H - 31)
        c.setFillColor(base.MUTED)
        c.setFont("Arial-Bold", 6.5)
        if logical % 2 == 0:
            c.drawString(base.MARGIN_X, base.PAGE_H - 23, "LEARNING STARK")
        else:
            c.drawRightString(base.PAGE_W - base.MARGIN_X, base.PAGE_H - 23, "CONTROL FLOW AND PATTERN MATCHING")
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

    frame = Frame(base.MARGIN_X, base.BOTTOM, base.CONTENT_W, base.PAGE_H - base.TOP - base.BOTTOM, id="main", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    doc = base.ChapterDocTemplate(
        str(OUTPUT),
        pagesize=(base.PAGE_W, base.PAGE_H),
        leftMargin=base.MARGIN_X,
        rightMargin=base.MARGIN_X,
        topMargin=base.TOP,
        bottomMargin=base.BOTTOM,
        title="Learning STARK - Chapter 4: Control Flow and Pattern Matching",
        author="STARK Language Project",
        subject="Core v1 conditional expressions, loops, control transfer, enums, patterns, exhaustive matching, and Option",
        keywords="STARK, programming language, Core v1, control flow, pattern matching, enums, Option, loops, exhaustiveness",
    )
    doc.addPageTemplates([
        base.PageTemplate(id="cover", frames=[frame], onPage=cover_page),
        base.PageTemplate(id="front", frames=[frame], onPage=base.front_page),
        base.PageTemplate(id="body", frames=[frame], onPage=body_page),
    ])

    legal = [
        Spacer(1, 250),
        Paragraph("<b>Learning STARK</b>", sty["legal"]),
        Paragraph("Chapter 4 preview: <i>Control Flow and Pattern Matching</i>", sty["legal"]),
        Spacer(1, 7),
        Paragraph("Early Access manuscript - July 2026", sty["legal"]),
        Paragraph("This chapter documents a language specification under active implementation. Core v1 is a normative draft and has not yet been validated by a conforming compiler.", sty["legal"]),
        Paragraph("The design and text are original to the STARK Language Project.", sty["legal"]),
        Spacer(1, 10),
        Paragraph("Source manuscript: <font face='CourierNew'>book/manuscript/ch04-control-flow-pattern-matching.md</font>", sty["legal"]),
    ]

    deck = "Control flow becomes dependable when every path has a type, every exit has a destination, and every state has a case."
    story: list[Flowable] = [
        Spacer(1, base.PAGE_H - base.TOP - base.BOTTOM - 1),
        NextPageTemplate("front"),
        PageBreak(),
        *legal,
        NextPageTemplate("body"),
        PageBreak(),
        base.ChapterOpener("Control Flow and Pattern Matching", deck, number="04"),
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
