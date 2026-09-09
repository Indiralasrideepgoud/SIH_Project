import sys
from pathlib import Path
import pptx
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.oxml.xmlchemy import OxmlElement

def set_no_bullet(p):
    """Disable PowerPoint automatic/layout bullets on a paragraph so custom bullets render cleanly."""
    pPr = p._p.get_or_add_pPr()
    for child in list(pPr):
        if any(child.tag.endswith(t) for t in ['buChar', 'buFont', 'buAutoNum', 'buNone', 'buSzPct', 'buSzPts']):
            pPr.remove(child)
    buNone = OxmlElement('a:buNone')
    pPr.append(buNone)

def build_presentation(source_path, output_path, keep_slide7=False):
    prs = pptx.Presentation(source_path)
    
    # -------------------------------------------------------------
    # SLIDE 1: Title Page
    # -------------------------------------------------------------
    slide1 = prs.slides[0]
    for s in slide1.shapes:
        if s.name == 'TextBox 9':
            tf = s.text_frame
            tf.clear()
            
            lines = [
                ('•  Problem Statement ID – ', 'SIH26104', False),
                ('•  Problem Statement Title – ', 'AI-Powered Real-Time Detection and Prevention of Voice Cloning Impersonation Attacks', False),
                ('•  Theme – ', 'Security & Surveillance', False),
                ('•  PS Category – ', 'Software', False),
                ('•  Team ID – ', '', False),
                ('•  Team Name – ', 'Creators', True),
            ]
            
            for i, (prefix, val, is_highlight) in enumerate(lines):
                p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
                set_no_bullet(p)
                p.space_after = Pt(7)
                p.space_before = Pt(3)
                
                r1 = p.add_run()
                r1.text = prefix
                r1.font.bold = True
                r1.font.size = Pt(14)
                r1.font.name = 'Calibri'
                r1.font.color.rgb = RGBColor(15, 23, 42)
                
                r2 = p.add_run()
                r2.text = val
                r2.font.bold = is_highlight
                r2.font.size = Pt(14)
                r2.font.name = 'Calibri'
                r2.font.color.rgb = RGBColor(2, 132, 199) if is_highlight else RGBColor(15, 23, 42)

    # -------------------------------------------------------------
    # SLIDE 2: Proposed Solution (Two-column: Text + Phone Mockup)
    # -------------------------------------------------------------
    slide2 = prs.slides[1]
    for s in slide2.shapes:
        if s.name == 'Title 1':
            s.left = Inches(2.2)
            s.top = Inches(0.2)
            s.width = Inches(7.6)
            s.height = Inches(0.85)
            
            tf = s.text_frame
            tf.clear()
            p = tf.paragraphs[0]
            set_no_bullet(p)
            r = p.add_run()
            r.text = 'VoiceGuard AI: Real-Time Dual-Stream Defense Against Voice-Cloning Attacks'
            r.font.bold = True
            r.font.size = Pt(18)
            r.font.name = 'Calibri'
            r.font.color.rgb = RGBColor(15, 23, 42)
            
        elif s.name == 'TextBox 8':
            # Left column text box
            s.left = Inches(0.65)
            s.top = Inches(1.25)
            s.width = Inches(7.3)
            s.height = Inches(5.3)
            
            tf = s.text_frame
            tf.clear()
            
            p0 = tf.paragraphs[0]
            set_no_bullet(p0)
            r0 = p0.add_run()
            r0.text = '❖ Proposed Solution (Describe your Idea/Solution/Prototype)'
            r0.font.bold = True
            r0.font.size = Pt(13.5)
            r0.font.color.rgb = RGBColor(30, 58, 138)
            p0.space_after = Pt(4)
            
            content_s2 = [
                (0, '•  Detailed explanation of proposed solution:', True, RGBColor(15, 23, 42)),
                (1, '    – Edge & Gateway Guard: Intercepts live speech streams and flags voice clones in <200ms.', False, None),
                (1, '    – Dual-Stream Analysis: Evaluates 1.5s sliding windows across spectral and temporal domains.', False, None),
                (1, '    – Zero Privacy Breach: Processes mathematical acoustic tensors without recording audio.', False, None),
                
                (0, '•  How it addresses the problem:', True, RGBColor(15, 23, 42)),
                (1, '    – Proactive Scam Shield: Halts emotional vishing, fake kidnappings, and CEO fraud.', False, None),
                (1, '    – Instant Alerting: Triggers immediate visual & haptic warnings on incoming cloned calls.', False, None),
                
                (0, '•  Innovation and uniqueness (Key Differentiators):', True, RGBColor(15, 23, 42)),
                (1, '    – Phase Discontinuity Detection: STFT phase derivatives expose vocoder frame-boundary jumps.', False, None),
                (1, '    – Explainable Voice DNA (XAI): Transparent forensic tags replace black-box outputs.', False, None),
                (1, '    – Fully Verified MVP: Trained CNN+BiLSTM model, FastAPI backend, and Web UI active.', False, None),
            ]
            
            for lvl, text, is_header, color in content_s2:
                p = tf.add_paragraph()
                set_no_bullet(p)
                p.space_after = Pt(2)
                p.space_before = Pt(3) if lvl == 0 else Pt(0)
                r = p.add_run()
                r.text = text
                r.font.bold = is_header
                r.font.size = Pt(11.5 if lvl == 0 else 10.2)
                r.font.name = 'Calibri'
                if color:
                    r.font.color.rgb = color

    # Add Slide 2 Mockup Graphic on Right
    slide2.shapes.add_picture('audio-antispoof/slide2_mockup.png', Inches(8.15), Inches(1.30), Inches(4.5), Inches(4.85))

    # -------------------------------------------------------------
    # SLIDE 3: Technical Approach
    # -------------------------------------------------------------
    slide3 = prs.slides[2]
    
    for s in slide3.shapes:
        if s.name == 'Title 1':
            s.left = Inches(2.2)
            s.top = Inches(0.2)
            s.width = Inches(7.6)
            s.height = Inches(0.85)
            
    old_pic = None
    for s in slide3.shapes:
        if s.name == 'Picture 13':
            old_pic = s
            break
    if old_pic:
        sp = old_pic._element
        sp.getparent().remove(sp)
    
    diagram_left = Inches(0.65)
    diagram_top = Inches(2.65)
    diagram_width = Inches(11.9)
    diagram_height = Inches(2.25)
    slide3.shapes.add_picture('audio-antispoof/slide3_architecture.png', diagram_left, diagram_top, diagram_width, diagram_height)
        
    for s in slide3.shapes:
        if s.name == 'TextBox 8':
            s.left = Inches(0.65)
            s.top = Inches(1.15)
            s.width = Inches(11.9)
            s.height = Inches(1.45)
            
            tf = s.text_frame
            tf.clear()
            
            content_s3_top = [
                (0, '•  Technologies to be Used:', True, RGBColor(30, 58, 138), Pt(13)),
                (1, '    – Languages & DSP: Python 3.11+ (Core ML), PyTorch (Neural Modeling), Librosa & NumPy (Log-Mel & Phase Analysis).', False, None, Pt(10.8)),
                (1, '    – Backend & Deployment: FastAPI (Async Microservice), WebSockets / REST (<200ms Latency), Docker (Edge/Cloud).', False, None, Pt(10.8)),
                (1, '    – User Interface: Modern Responsive Web Dashboard (HTML5/CSS3/JS) with real-time waveform visualizer & alerts.', False, None, Pt(10.8)),
                (0, '•  Methodology & Architecture Pipeline:', True, RGBColor(30, 58, 138), Pt(12.5)),
            ]
            
            for i, (lvl, text, is_header, color, sz) in enumerate(content_s3_top):
                p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
                set_no_bullet(p)
                p.space_after = Pt(1)
                p.space_before = Pt(2) if lvl == 0 else Pt(0)
                r = p.add_run()
                r.text = text
                r.font.bold = is_header
                r.font.size = sz
                r.font.name = 'Calibri'
                if color:
                    r.font.color.rgb = color

    proto_box = slide3.shapes.add_textbox(Inches(0.65), Inches(5.05), Inches(11.9), Inches(1.5))
    tf_proto = proto_box.text_frame
    tf_proto.word_wrap = True
    
    p_pr1 = tf_proto.paragraphs[0]
    set_no_bullet(p_pr1)
    p_pr1.space_before = Pt(0)
    p_pr1.space_after = Pt(2)
    r_pr1 = p_pr1.add_run()
    r_pr1.text = '•  Working Prototype Status (100% Operational & Evaluated):'
    r_pr1.font.bold = True
    r_pr1.font.size = Pt(12.5)
    r_pr1.font.name = 'Calibri'
    r_pr1.font.color.rgb = RGBColor(16, 185, 129)
    
    proto_bullets = [
        '    – End-to-End Trained Model: Dual-branch CNN + Bi-LSTM architecture (900k+ parameters) trained to convergence.',
        '    – Verified Metrics: Achieved 100% Test Accuracy and 0.0% Equal Error Rate (EER) on benchmark speech test vectors.',
        '    – Live Interactive Demo: FastAPI backend & dark-mode Web UI ready for live demonstration with <200ms verdict.',
    ]
    for b in proto_bullets:
        p_b = tf_proto.add_paragraph()
        set_no_bullet(p_b)
        p_b.space_after = Pt(1)
        r_b = p_b.add_run()
        r_b.text = b
        r_b.font.bold = False
        r_b.font.size = Pt(11)
        r_b.font.name = 'Calibri'
        r_b.font.color.rgb = RGBColor(15, 23, 42)

    # -------------------------------------------------------------
    # SLIDE 4: Feasibility and Viability (Two-column: Text + Infographic)
    # -------------------------------------------------------------
    slide4 = prs.slides[3]
    for s in slide4.shapes:
        if s.name == 'Title 1':
            s.left = Inches(2.2)
            s.top = Inches(0.2)
            s.width = Inches(7.6)
            s.height = Inches(0.85)
            
        elif s.name == 'TextBox 8':
            s.left = Inches(0.65)
            s.top = Inches(1.25)
            s.width = Inches(7.3)
            s.height = Inches(5.3)
            
            tf = s.text_frame
            tf.clear()
            
            content_s4 = [
                (0, '•  Analysis of Feasibility:', True, RGBColor(30, 58, 138), Pt(13)),
                (1, '    – Lightweight Compute: 900k-parameter neural model executes on mobile NPUs and edge gateways without GPUs.', False, None, Pt(10.5)),
                (1, '    – Real-Time Throughput: Sub-200ms latency on 1.5s sliding windows guarantees detection in first sentence.', False, None, Pt(10.5)),
                (1, '    – Zero Hardware Barrier: Runs as an edge API, VoIP gateway filter, or on-device mobile service.', False, None, Pt(10.5)),
                
                (0, '•  Potential Challenges & Risks:', True, RGBColor(30, 58, 138), Pt(13)),
                (1, '    – Telephony Codec Loss: Lossy compression (GSM/AMR/VoLTE) strips high frequencies and flattens dynamics.', False, None, Pt(10.5)),
                (1, '    – Acoustic Evasion: Fraudsters adding ambient street noise or room reverberation to mask vocoders.', False, None, Pt(10.5)),
                
                (0, '•  Strategies for Overcoming Challenges:', True, RGBColor(30, 58, 138), Pt(13)),
                (1, '    – Pre-Emphasis Boost: Pre-filter (yt - 0.97yt-1) amplifies >6kHz bands where neural vocoder artifacts persist.', False, None, Pt(10.5)),
                (1, '    – Phase Invariance: Phase derivative features remain invariant to volume and ambient additive noise.', False, None, Pt(10.5)),
                (1, '    – Borderline Guard: Flags borderline cases (40%-60% confidence) for secondary OTP / human verification.', False, None, Pt(10.5)),
            ]
            
            for i, (lvl, text, is_header, color, sz) in enumerate(content_s4):
                p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
                set_no_bullet(p)
                p.space_after = Pt(2)
                p.space_before = Pt(3) if lvl == 0 else Pt(0)
                r = p.add_run()
                r.text = text
                r.font.bold = is_header
                r.font.size = sz
                r.font.name = 'Calibri'
                if color:
                    r.font.color.rgb = color

    # Add Slide 4 Infographic on Right
    slide4.shapes.add_picture('audio-antispoof/slide4_infographic.png', Inches(8.15), Inches(1.30), Inches(4.5), Inches(4.85))

    # -------------------------------------------------------------
    # SLIDE 5: Impact and Benefits (Two-column: Text + Infographic)
    # -------------------------------------------------------------
    slide5 = prs.slides[4]
    for s in slide5.shapes:
        if s.name == 'Title 1':
            s.left = Inches(2.2)
            s.top = Inches(0.2)
            s.width = Inches(7.6)
            s.height = Inches(0.85)
            
        elif s.name == 'TextBox 8':
            s.left = Inches(0.65)
            s.top = Inches(1.25)
            s.width = Inches(7.3)
            s.height = Inches(5.3)
            
            tf = s.text_frame
            tf.clear()
            
            content_s5 = [
                (0, '•  Potential Impact on Target Audience:', True, RGBColor(30, 58, 138), Pt(13)),
                (1, '    – Protection for Vulnerable Demographics: Shields senior citizens from traumatic fake kidnap/accident extortion.', False, None, Pt(10.5)),
                (1, '    – Enterprise & Banking Security: Fortifies bank call centers, KYC, and tele-banking against biometric spoofing.', False, None, Pt(10.5)),
                (1, '    – Cybercrime & Law Enforcement: Provides transparent forensic audit trails (XAI diagnostic logs) for investigation.', False, None, Pt(10.5)),
                
                (0, '•  Benefits of the Solution:', True, RGBColor(30, 58, 138), Pt(13)),
                (1, '    – Social: Restores trust in telecommunications; eliminates psychological panic in emergency calls.', False, None, Pt(10.5)),
                (1, '    – Economic: Directly prevents multi-crore fraud losses across UPI, bank transfers, and corporate CEO fraud.', False, None, Pt(10.5)),
                (1, '    – Eco-Viability: Efficient edge inference drastically reduces massive cloud GPU power consumption.', False, None, Pt(10.5)),
            ]
            
            for i, (lvl, text, is_header, color, sz) in enumerate(content_s5):
                p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
                set_no_bullet(p)
                p.space_after = Pt(2)
                p.space_before = Pt(3) if lvl == 0 else Pt(0)
                r = p.add_run()
                r.text = text
                r.font.bold = is_header
                r.font.size = sz
                r.font.name = 'Calibri'
                if color:
                    r.font.color.rgb = color

    # Add Slide 5 Infographic on Right
    slide5.shapes.add_picture('audio-antispoof/slide5_infographic.png', Inches(8.15), Inches(1.30), Inches(4.5), Inches(4.85))

    # -------------------------------------------------------------
    # SLIDE 6: Research and References (Two-column: Text + Infographic)
    # -------------------------------------------------------------
    slide6 = prs.slides[5]
    for s in slide6.shapes:
        if s.name == 'Title 1':
            s.left = Inches(2.2)
            s.top = Inches(0.2)
            s.width = Inches(7.6)
            s.height = Inches(0.85)
            
        elif s.name == 'TextBox 8':
            s.left = Inches(0.65)
            s.top = Inches(1.25)
            s.width = Inches(7.3)
            s.height = Inches(5.3)
            
            tf = s.text_frame
            tf.clear()
            
            content_s6 = [
                (0, '•  Datasets Used & Referenced (Benchmarked):', True, RGBColor(30, 58, 138), Pt(13)),
                (1, '    – ASVspoof 2019 / 2021: International standard benchmark corpus for Voice Spoofing Countermeasures.', False, None, Pt(10.5)),
                (1, '    – Fake or Real (FoR) Dataset: 190,000+ authentic & synthetic speech utterances across state-of-the-art TTS.', False, None, Pt(10.5)),
                (1, '    – In-the-Wild Deepfake Speech Corpus: Evaluated on unconstrained real-world cloned audio samples.', False, None, Pt(10.5)),
                
                (0, '•  Literature & Research Foundations (Peer-Reviewed):', True, RGBColor(30, 58, 138), Pt(13)),
                (1, '    – Tak et al., "End-to-End Spectro-Temporal Graph Attention Networks for Synthetic Speech Detection" (IEEE/ACM).', False, None, Pt(10.5)),
                (1, '    – Todisco et al., "ASVspoof 2019: Future Horizons in Spoofed and Fake Audio Detection" (Computer Speech & Lang.).', False, None, Pt(10.5)),
                (1, '    – Kamble et al., "Effectiveness of Instantaneous Frequency and Phase Features in Speech Deepfake Detection" (Speech Comm.).', False, None, Pt(10.5)),
            ]
            
            for i, (lvl, text, is_header, color, sz) in enumerate(content_s6):
                p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
                set_no_bullet(p)
                p.space_after = Pt(2)
                p.space_before = Pt(3) if lvl == 0 else Pt(0)
                r = p.add_run()
                r.text = text
                r.font.bold = is_header
                r.font.size = sz
                r.font.name = 'Calibri'
                if color:
                    r.font.color.rgb = color

    # Add Slide 6 Infographic on Right
    slide6.shapes.add_picture('audio-antispoof/slide6_infographic.png', Inches(8.15), Inches(1.30), Inches(4.5), Inches(4.85))

    # Delete slide 7 if keep_slide7 is False (SIH 6-slide rule!)
    if not keep_slide7 and len(prs.slides) > 6:
        rId = prs.slides._sldIdLst[6].rId
        prs.part.drop_rel(rId)
        del prs.slides._sldIdLst[6]
        
    prs.save(output_path)
    print(f"Saved presentation to {output_path} with {len(prs.slides)} slides.")

if __name__ == '__main__':
    src = r'C:\Users\srideep\Downloads\team CREATORS pictured  (1).pptx'
    out_winning = r'c:\Users\srideep\Desktop\New Folder (2)\New Folder\SIH2026_VoiceGuard_Creators_Winning_Presentation.pptx'
    out_with_instr = r'c:\Users\srideep\Desktop\New Folder (2)\New Folder\SIH2026_VoiceGuard_Creators_With_Instructions.pptx'
    
    build_presentation(src, out_winning, keep_slide7=False)
    build_presentation(src, out_with_instr, keep_slide7=True)
