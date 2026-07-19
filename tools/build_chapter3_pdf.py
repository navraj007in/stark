#!/usr/bin/env python3
"""Build Chapter 3 of Learning STARK using the established book design."""

from __future__ import annotations

import sys
from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.platypus import Flowable, Frame, NextPageTemplate, PageBreak, Paragraph, Spacer

import build_chapter1_pdf as base


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "book/manuscript/ch03-values-types-expressions.md"
OUTPUT = ROOT / "output/pdf/learning-stark-chapter-03.pdf"


class TypeFamiliesDiagram(base.Diagram):
    caption = "Figure 3-1. Every expression has one static type drawn from a small set of composable families."

    def __init__(self): super().__init__(154)

    def draw(self):
        c = self.canv; c.saveState()
        base.Diagram._box(self, c, 148, 58, 104, 40, "STATIC TYPE", fill=base.INK, stroke=base.INK, font=7)
        items = [(0,105,"PRIMITIVE","Int32 / Bool"),(102,105,"COMPOSITE","array / tuple"),(204,105,"REFERENCE","&amp;T / &amp;mut T"),(306,105,"FUNCTION","fn(A) -&gt; B"),(153,8,"GENERIC","T / Tensor&lt;T&gt;")]
        for x,y,label,detail in items:
            base.Diagram._box(self,c,x,y,94,34,f"{label}<br/><font size='5.5'>{detail}</font>",fill=base.BLUE_PALE,stroke=base.BLUE,font=6)
            x2=x+47; y2=y+34 if y<50 else y
            base.Diagram._arrow(self,c,x2,y2,200,58 if y<50 else 98,base.GRID)
        c.restoreState()


class LiteralDefaultsDiagram(base.Diagram):
    caption = "Figure 3-2. Defaults classify unsuffixed literals; suffixes make representation explicit."

    def __init__(self): super().__init__(142)

    def draw(self):
        c=self.canv; c.saveState(); gap=10; w=(base.CONTENT_W-gap*2)/3
        data=[("INTEGER","42","Int32 if it fits\notherwise Int64",base.BLUE_PALE,base.BLUE),("FLOAT","3.14","Float64",HexColor("#E7ECE8"),base.GREEN),("SUFFIXED","42u64","selected type: UInt64",HexColor("#F2E6D7"),base.RUST)]
        for i,(lab,lit,res,fill,accent) in enumerate(data):
            x=i*(w+gap); c.setFillColor(fill); c.setStrokeColor(base.GRID); c.roundRect(x,25,w,92,6,fill=1,stroke=1)
            c.setFillColor(accent); c.rect(x,25,5,92,fill=1,stroke=0); c.setFont("Arial-Bold",6); c.drawString(x+14,100,lab)
            c.setFillColor(base.INK); c.setFont("CourierNew-Bold",12); c.drawCentredString(x+w/2,72,lit)
            c.setFillColor(base.MUTED); c.setFont("Georgia",6.4)
            for j,line in enumerate(res.split("\n")): c.drawCentredString(x+w/2,48-j*9,line)
        c.restoreState()


class ArraySliceDiagram(base.Diagram):
    caption = "Figure 3-3. Borrowing a fixed array as a slice preserves element type while moving length to runtime."

    def __init__(self): super().__init__(132)

    def draw(self):
        c=self.canv; c.saveState(); y=53
        base.Diagram._box(self,c,0,y,145,45,"[Int32; 5]<br/><font size='6'>length in the type</font>",fill=base.BLUE_PALE,stroke=base.BLUE,font=7)
        base.Diagram._arrow(self,c,154,y+22,242,y+22,base.RUST); c.setFillColor(base.RUST); c.setFont("CourierNew-Bold",7); c.drawCentredString(198,y+32,"&amp;fixed")
        base.Diagram._box(self,c,250,y,150,45,"&amp;[Int32]<br/><font size='6'>runtime length view</font>",fill=HexColor("#E7ECE8"),stroke=base.GREEN,font=7)
        c.setFillColor(base.MUTED); c.setFont("Arial",6.3); c.drawCentredString(200,30,"same storage - shared access - no element copy")
        c.restoreState()


class PrecedenceDiagram(base.Diagram):
    caption = "Figure 3-4. Major precedence bands, from tightest binding at left to loosest at right."

    def __init__(self): super().__init__(132)

    def draw(self):
        c=self.canv; c.saveState(); labels=[("POSTFIX","call [ ] . ?"),("UNARY / CAST","! - &amp; / as"),("POWER / PRODUCT","** then * / %"),("SUM / SHIFT","+ - then &lt;&lt; &gt;&gt;"),("COMPARE / LOGIC","&lt; == then &amp;&amp; ||"),("RANGE / ASSIGN",".. ..= then =")]
        gap=5; w=(base.CONTENT_W-gap*5)/6; y=44
        for i,(a,b) in enumerate(labels):
            x=i*(w+gap); fill=base.BLUE_PALE if i<2 else (HexColor("#E7ECE8") if i<4 else HexColor("#F2E6D7")); base.Diagram._box(self,c,x,y,w,50,f"{a}<br/><font size='5.2'>{b}</font>",fill=fill,font=5.6)
            if i<5: base.Diagram._arrow(self,c,x+w+1,y+25,x+w+gap-1,y+25,base.GRID)
        c.setFillColor(base.MUTED); c.setFont("Arial",6.2); c.drawCentredString(200,27,"tightest binding  ->  loosest binding")
        c.restoreState()


class SafetyBoundaryDiagram(base.Diagram):
    caption = "Figure 3-5. Static rejection handles contradictions; traps guard data-dependent failures."

    def __init__(self): super().__init__(150)

    def draw(self):
        c=self.canv; c.saveState(); gap=18; w=(base.CONTENT_W-gap)/2
        panels=[(0,"COMPILE TIME","type mismatch\ninvalid place\nknown bad index",base.BLUE_PALE,base.BLUE),(w+gap,"RUNTIME TRAP","overflow / divide by zero\ndynamic bounds failure\ninvalid numeric cast",HexColor("#F2E6D7"),base.RUST)]
        for x,title,lines,fill,accent in panels:
            c.setFillColor(fill); c.setStrokeColor(base.GRID); c.roundRect(x,28,w,98,6,fill=1,stroke=1); c.setFillColor(accent); c.rect(x,28,5,98,fill=1,stroke=0); c.setFont("Arial-Bold",6.5); c.drawString(x+16,108,title); c.setFont("Georgia",7); c.setFillColor(base.INK)
            for j,line in enumerate(lines.split("\n")): c.drawString(x+16,84-j*18,"- "+line)
        c.restoreState()


class TypeTraceDiagram(base.Diagram):
    caption = "Figure 3-6. Type-checking proceeds from leaves to operators until the whole condition is Bool."

    def __init__(self): super().__init__(140)

    def draw(self):
        c=self.canv; c.saveState()
        base.Diagram._box(self,c,15,82,145,38,"samples &gt; 0u64<br/><font size='6'>Bool</font>",fill=base.BLUE_PALE,stroke=base.BLUE,font=7)
        base.Diagram._box(self,c,240,82,145,38,"ratio &gt; 1.0<br/><font size='6'>Bool</font>",fill=HexColor("#E7ECE8"),stroke=base.GREEN,font=7)
        base.Diagram._arrow(self,c,88,80,180,54,base.GRID); base.Diagram._arrow(self,c,313,80,220,54,base.GRID)
        base.Diagram._box(self,c,155,22,90,36,"&amp;&amp;<br/><font size='6'>Bool</font>",fill=base.INK,stroke=base.INK,font=7)
        c.restoreState()


CHAPTER_DIAGRAMS={"TYPE_FAMILIES":TypeFamiliesDiagram,"LITERAL_DEFAULTS":LiteralDefaultsDiagram,"ARRAY_SLICE":ArraySliceDiagram,"PRECEDENCE":PrecedenceDiagram,"SAFETY_BOUNDARY":SafetyBoundaryDiagram,"TYPE_TRACE":TypeTraceDiagram}


def cover_page(c,doc):
    c.saveState(); c.setFillColor(base.CODE_BG); c.rect(0,0,base.PAGE_W,base.PAGE_H,fill=1,stroke=0); base.draw_topographic(c,base.PAGE_W*.42,base.PAGE_H*.11,base.PAGE_W*.68,base.PAGE_H*.78)
    c.setFillColor(base.RUST); c.rect(0,base.PAGE_H-14,base.PAGE_W,14,fill=1,stroke=0); c.setFont("Arial-Bold",8); c.setFillColor(HexColor("#D9A38F")); c.drawString(base.MARGIN_X,base.PAGE_H-55,"EARLY ACCESS / CORE V1 SPECIFICATION EDITION")
    c.setFont("Georgia-Bold",37); c.setFillColor(base.WHITE); c.drawString(base.MARGIN_X,base.PAGE_H-132,"Learning"); c.setFillColor(base.RUST); c.drawString(base.MARGIN_X,base.PAGE_H-173,"STARK")
    c.setStrokeColor(HexColor("#5A6466")); c.line(base.MARGIN_X,base.PAGE_H-198,base.PAGE_W-base.MARGIN_X,base.PAGE_H-198); c.setFont("Georgia-Italic",13.2); c.setFillColor(HexColor("#DDD7CB")); c.drawString(base.MARGIN_X,base.PAGE_H-224,"Safe Systems Programming for Typed ML Deployment")
    c.setFont("Arial-Bold",8.5); c.setFillColor(base.WHITE); c.drawString(base.MARGIN_X,84,"CHAPTER 03"); c.setFont("Georgia-Bold",16); c.drawString(base.MARGIN_X,59,"Values, Types, and Expressions"); c.setFont("Arial",7.2); c.setFillColor(HexColor("#AEB7B5")); c.drawRightString(base.PAGE_W-base.MARGIN_X,32,"STARK LANGUAGE PROJECT / JULY 2026"); c.restoreState()


def body_page(c,doc):
    c.saveState(); c.setFillColor(base.PAPER); c.rect(0,0,base.PAGE_W,base.PAGE_H,fill=1,stroke=0); logical=c.getPageNumber()-2
    if logical>1:
        c.setStrokeColor(base.GRID); c.setLineWidth(.45); c.line(base.MARGIN_X,base.PAGE_H-31,base.PAGE_W-base.MARGIN_X,base.PAGE_H-31); c.setFillColor(base.MUTED); c.setFont("Arial-Bold",6.5)
        if logical%2==0: c.drawString(base.MARGIN_X,base.PAGE_H-23,"LEARNING STARK")
        else: c.drawRightString(base.PAGE_W-base.MARGIN_X,base.PAGE_H-23,"VALUES, TYPES, AND EXPRESSIONS")
    c.setFillColor(base.MUTED); c.setFont("Arial",7); c.drawCentredString(base.PAGE_W/2,25,str(logical)); c.restoreState()


def build():
    base.register_fonts(); sty=base.styles(); OUTPUT.parent.mkdir(parents=True,exist_ok=True); base.DIAGRAMS.update(CHAPTER_DIAGRAMS); body=base.parse_markdown(SOURCE.read_text(encoding="utf-8"),sty)
    frame=Frame(base.MARGIN_X,base.BOTTOM,base.CONTENT_W,base.PAGE_H-base.TOP-base.BOTTOM,id="main",leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)
    doc=base.ChapterDocTemplate(str(OUTPUT),pagesize=(base.PAGE_W,base.PAGE_H),leftMargin=base.MARGIN_X,rightMargin=base.MARGIN_X,topMargin=base.TOP,bottomMargin=base.BOTTOM,title="Learning STARK - Chapter 3: Values, Types, and Expressions",author="STARK Language Project",subject="Core v1 values, scalar and composite types, inference, conversion, expressions, and runtime traps",keywords="STARK, programming language, Core v1, types, literals, arrays, slices, inference, casts, expressions")
    doc.addPageTemplates([base.PageTemplate(id="cover",frames=[frame],onPage=cover_page),base.PageTemplate(id="front",frames=[frame],onPage=base.front_page),base.PageTemplate(id="body",frames=[frame],onPage=body_page)])
    legal=[Spacer(1,250),Paragraph("<b>Learning STARK</b>",sty["legal"]),Paragraph("Chapter 3 preview: <i>Values, Types, and Expressions</i>",sty["legal"]),Spacer(1,7),Paragraph("Early Access manuscript - July 2026",sty["legal"]),Paragraph("This chapter documents a language specification under active implementation. Core v1 is a normative draft and has not yet been validated by a conforming compiler.",sty["legal"]),Paragraph("The design and text are original to the STARK Language Project.",sty["legal"]),Spacer(1,10),Paragraph("Source manuscript: <font face='CourierNew'>book/manuscript/ch03-values-types-expressions.md</font>",sty["legal"])]
    deck="Types are not labels added after a program is written. They are the vocabulary STARK uses to prove what each expression can mean."
    story=[Spacer(1,base.PAGE_H-base.TOP-base.BOTTOM-1),NextPageTemplate("front"),PageBreak(),*legal,NextPageTemplate("body"),PageBreak(),base.ChapterOpener("Values, Types, and Expressions",deck,number="03"),*body]
    doc.build(story); print(OUTPUT)


if __name__=="__main__":
    try: build()
    except Exception as exc: print(f"error: {exc}",file=sys.stderr); raise
