#!/usr/bin/env python3
"""Build Chapter 5 of Learning STARK using the established book design."""

from __future__ import annotations

import sys
from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.platypus import Flowable, Frame, NextPageTemplate, PageBreak, Paragraph, Spacer

import build_chapter1_pdf as base


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "book/manuscript/ch05-functions-structs-enums-methods.md"
OUTPUT = ROOT / "output/pdf/learning-stark-chapter-05.pdf"


class FunctionContractDiagram(base.Diagram):
    caption = "Figure 5-1. A function signature creates an explicit contract between caller and implementation."

    def __init__(self): super().__init__(142)

    def draw(self):
        c=self.canv; c.saveState()
        base.Diagram._box(self,c,0,52,105,46,"CALLER<br/><font size='6'>arguments</font>",fill=base.PAPER_DARK,font=7)
        base.Diagram._arrow(self,c,112,75,158,75,base.BLUE)
        base.Diagram._box(self,c,165,38,135,74,"SIGNATURE<br/><font size='6'>names + parameter types<br/>return type</font>",fill=base.BLUE_PALE,stroke=base.BLUE,font=7)
        base.Diagram._arrow(self,c,307,75,353,75,base.GREEN)
        base.Diagram._box(self,c,360,52,40,46,"BODY<br/><font size='6'>checked</font>",fill=HexColor("#E7ECE8"),stroke=base.GREEN,font=6.2)
        c.setFillColor(base.MUTED); c.setFont("Arial",6.2); c.drawCentredString(200,22,"local inference stops at the signature boundary")
        c.restoreState()


class ParameterModesDiagram(base.Diagram):
    caption = "Figure 5-2. Parameter form states whether a call transfers, shares, or exclusively borrows access."

    def __init__(self): super().__init__(150)

    def draw(self):
        c=self.canv; c.saveState(); gap=10; w=(base.CONTENT_W-2*gap)/3
        data=[("T","OWN","move or Copy",base.PAPER_DARK,base.GOLD),("&amp;T","SHARE","inspect",base.BLUE_PALE,base.BLUE),("&amp;mut T","EXCLUSIVE","modify",HexColor("#F2E6D7"),base.RUST)]
        for i,(code,label,detail,fill,accent) in enumerate(data):
            x=i*(w+gap); c.setFillColor(fill); c.setStrokeColor(base.GRID); c.roundRect(x,28,w,94,6,fill=1,stroke=1); c.setFillColor(accent); c.rect(x,28,5,94,fill=1,stroke=0)
            c.setFont("CourierNew-Bold",11); c.drawCentredString(x+w/2,91,code); c.setFont("Arial-Bold",6.2); c.drawCentredString(x+w/2,67,label); c.setFillColor(base.MUTED); c.setFont("Georgia",6.5); c.drawCentredString(x+w/2,46,detail)
        c.restoreState()


class ProductSumDiagram(base.Diagram):
    caption = "Figure 5-3. Struct fields coexist; enum variants are mutually exclusive alternatives."

    def __init__(self): super().__init__(154)

    def draw(self):
        c=self.canv; c.saveState(); gap=18; w=(base.CONTENT_W-gap)/2
        c.setFillColor(base.BLUE_PALE); c.setStrokeColor(base.BLUE); c.roundRect(0,28,w,101,7,fill=1,stroke=1); c.setFillColor(base.BLUE); c.setFont("Arial-Bold",6.5); c.drawString(15,110,"STRUCT / PRODUCT")
        for i,t in enumerate(["layout","resize","normalize"]): base.Diagram._box(self,c,15,80-i*22,w-30,17,t,fill=base.WHITE,stroke=base.GRID,font=6)
        x=w+gap; c.setFillColor(HexColor("#F2E6D7")); c.setStrokeColor(base.RUST); c.roundRect(x,28,w,101,7,fill=1,stroke=1); c.setFillColor(base.RUST); c.setFont("Arial-Bold",6.5); c.drawString(x+15,110,"ENUM / SUM")
        for i,t in enumerate(["None","Exact(...) ","Fit { ... }"]):
            base.Diagram._box(self,c,x+15,80-i*22,w-30,17,t,fill=base.WHITE,stroke=base.GRID,font=6)
        c.restoreState()


class ReceiverModesDiagram(base.Diagram):
    caption = "Figure 5-4. The receiver declares whether a method consumes, observes, or mutates its value."

    def __init__(self): super().__init__(142)

    def draw(self):
        c=self.canv; c.saveState(); labels=[("self","CONSUME","receiver moved",base.PAPER_DARK,base.GOLD),("&amp;self","OBSERVE","shared borrow",base.BLUE_PALE,base.BLUE),("&amp;mut self","MODIFY","exclusive borrow",HexColor("#F2E6D7"),base.RUST)]; gap=10; w=(base.CONTENT_W-gap*2)/3
        for i,(code,title,detail,fill,accent) in enumerate(labels):
            x=i*(w+gap); base.Diagram._box(self,c,x,38,w,75,f"<font face='CourierNew' size='9'>{code}</font><br/><font size='6'>{title}</font><br/><font size='5.5'>{detail}</font>",fill=fill,stroke=accent,font=7)
        c.restoreState()


class MethodResolutionDiagram(base.Diagram):
    caption = "Figure 5-5. Dot-call resolution applies a small, defined search rather than arbitrary dispatch."

    def __init__(self): super().__init__(130)

    def draw(self):
        c=self.canv; c.saveState(); labels=["DOT CALL","INHERENT","TRAIT","AUTO-BORROW","AUTO-DEREF"]
        gap=6; w=(base.CONTENT_W-gap*4)/5; y=49
        for i,label in enumerate(labels):
            fill=base.PAPER_DARK if i==0 else (base.BLUE_PALE if i<3 else HexColor("#E7ECE8")); base.Diagram._box(self,c,i*(w+gap),y,w,40,label,fill=fill,stroke=base.GRID,font=5.8)
            if i<4: base.Diagram._arrow(self,c,i*(w+gap)+w+1,y+20,i*(w+gap)+w+gap-1,y+20,base.BLUE)
        c.setFillColor(base.MUTED); c.setFont("Arial",6.1); c.drawCentredString(200,29,"visibility and ambiguity checks complete the decision")
        c.restoreState()


class ConfigApiDiagram(base.Diagram):
    caption = "Figure 5-6. The image configuration API separates construction, mutation, inspection, and transfer."

    def __init__(self): super().__init__(158)

    def draw(self):
        c=self.canv; c.saveState()
        base.Diagram._box(self,c,145,58,110,45,"ImageConfig<br/><font size='6'>owned value</font>",fill=base.INK,stroke=base.INK,font=7)
        nodes=[(0,105,"new(...) -&gt; Self",base.BLUE_PALE,base.BLUE),(290,105,"channel_count(&amp;self)",HexColor("#E7ECE8"),base.GREEN),(0,12,"with_resize(&amp;mut self)",HexColor("#F2E6D7"),base.RUST),(290,12,"return config",base.PAPER_DARK,base.GOLD)]
        for x,y,textv,fill,stroke in nodes: base.Diagram._box(self,c,x,y,110,34,textv,fill=fill,stroke=stroke,font=5.9)
        base.Diagram._arrow(self,c,112,118,147,94,base.BLUE); base.Diagram._arrow(self,c,254,94,288,118,base.GREEN); base.Diagram._arrow(self,c,112,29,147,63,base.RUST); base.Diagram._arrow(self,c,254,63,288,29,base.GOLD)
        c.restoreState()


CHAPTER_DIAGRAMS={"FUNCTION_CONTRACT":FunctionContractDiagram,"PARAMETER_MODES":ParameterModesDiagram,"PRODUCT_SUM":ProductSumDiagram,"RECEIVER_MODES":ReceiverModesDiagram,"METHOD_RESOLUTION":MethodResolutionDiagram,"CONFIG_API":ConfigApiDiagram}


def cover_page(c,doc):
    c.saveState(); c.setFillColor(base.CODE_BG); c.rect(0,0,base.PAGE_W,base.PAGE_H,fill=1,stroke=0); base.draw_topographic(c,base.PAGE_W*.42,base.PAGE_H*.11,base.PAGE_W*.68,base.PAGE_H*.78)
    c.setFillColor(base.RUST); c.rect(0,base.PAGE_H-14,base.PAGE_W,14,fill=1,stroke=0); c.setFont("Arial-Bold",8); c.setFillColor(HexColor("#D9A38F")); c.drawString(base.MARGIN_X,base.PAGE_H-55,"EARLY ACCESS / CORE V1 SPECIFICATION EDITION")
    c.setFont("Georgia-Bold",37); c.setFillColor(base.WHITE); c.drawString(base.MARGIN_X,base.PAGE_H-132,"Learning"); c.setFillColor(base.RUST); c.drawString(base.MARGIN_X,base.PAGE_H-173,"STARK")
    c.setStrokeColor(HexColor("#5A6466")); c.line(base.MARGIN_X,base.PAGE_H-198,base.PAGE_W-base.MARGIN_X,base.PAGE_H-198); c.setFont("Georgia-Italic",13.2); c.setFillColor(HexColor("#DDD7CB")); c.drawString(base.MARGIN_X,base.PAGE_H-224,"Safe Systems Programming for Typed ML Deployment")
    c.setFont("Arial-Bold",8.5); c.setFillColor(base.WHITE); c.drawString(base.MARGIN_X,84,"CHAPTER 05"); c.setFont("Georgia-Bold",15.2); c.drawString(base.MARGIN_X,59,"Functions, Structs, Enums, and Methods"); c.setFont("Arial",7.2); c.setFillColor(HexColor("#AEB7B5")); c.drawRightString(base.PAGE_W-base.MARGIN_X,32,"STARK LANGUAGE PROJECT / JULY 2026"); c.restoreState()


def body_page(c,doc):
    c.saveState(); c.setFillColor(base.PAPER); c.rect(0,0,base.PAGE_W,base.PAGE_H,fill=1,stroke=0); logical=c.getPageNumber()-2
    if logical>1:
        c.setStrokeColor(base.GRID); c.setLineWidth(.45); c.line(base.MARGIN_X,base.PAGE_H-31,base.PAGE_W-base.MARGIN_X,base.PAGE_H-31); c.setFillColor(base.MUTED); c.setFont("Arial-Bold",6.5)
        if logical%2==0: c.drawString(base.MARGIN_X,base.PAGE_H-23,"LEARNING STARK")
        else: c.drawRightString(base.PAGE_W-base.MARGIN_X,base.PAGE_H-23,"FUNCTIONS, STRUCTS, ENUMS, AND METHODS")
    c.setFillColor(base.MUTED); c.setFont("Arial",7); c.drawCentredString(base.PAGE_W/2,25,str(logical)); c.restoreState()


def build():
    base.register_fonts(); sty=base.styles(); OUTPUT.parent.mkdir(parents=True,exist_ok=True); base.DIAGRAMS.update(CHAPTER_DIAGRAMS); body=base.parse_markdown(SOURCE.read_text(encoding="utf-8"),sty)
    frame=Frame(base.MARGIN_X,base.BOTTOM,base.CONTENT_W,base.PAGE_H-base.TOP-base.BOTTOM,id="main",leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)
    doc=base.ChapterDocTemplate(str(OUTPUT),pagesize=(base.PAGE_W,base.PAGE_H),leftMargin=base.MARGIN_X,rightMargin=base.MARGIN_X,topMargin=base.TOP,bottomMargin=base.BOTTOM,title="Learning STARK - Chapter 5: Functions, Structs, Enums, and Methods",author="STARK Language Project",subject="Core v1 functions, ownership-aware parameters, structs, enums, associated functions, methods, receivers, and Self",keywords="STARK, programming language, Core v1, functions, structs, enums, methods, receivers, Self, API design")
    doc.addPageTemplates([base.PageTemplate(id="cover",frames=[frame],onPage=cover_page),base.PageTemplate(id="front",frames=[frame],onPage=base.front_page),base.PageTemplate(id="body",frames=[frame],onPage=body_page)])
    legal=[Spacer(1,250),Paragraph("<b>Learning STARK</b>",sty["legal"]),Paragraph("Chapter 5 preview: <i>Functions, Structs, Enums, and Methods</i>",sty["legal"]),Spacer(1,7),Paragraph("Early Access manuscript - July 2026",sty["legal"]),Paragraph("This chapter documents a language specification under active implementation. Core v1 is a normative draft and has not yet been validated by a conforming compiler.",sty["legal"]),Paragraph("The design and text are original to the STARK Language Project.",sty["legal"]),Spacer(1,10),Paragraph("Source manuscript: <font face='CourierNew'>book/manuscript/ch05-functions-structs-enums-methods.md</font>",sty["legal"])]
    deck="A production API makes data shape, ownership, construction, and behavior visible before the function body is read."
    story:list[Flowable]=[Spacer(1,base.PAGE_H-base.TOP-base.BOTTOM-1),NextPageTemplate("front"),PageBreak(),*legal,NextPageTemplate("body"),PageBreak(),base.ChapterOpener("Functions, Structs, Enums, and Methods",deck,number="05"),*body]
    doc.build(story); print(OUTPUT)


if __name__=="__main__":
    try: build()
    except Exception as exc: print(f"error: {exc}",file=sys.stderr); raise
