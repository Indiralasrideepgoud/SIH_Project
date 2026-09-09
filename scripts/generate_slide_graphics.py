import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def generate_slide2_mockup():
    fig, ax = plt.subplots(figsize=(4.6, 4.9), dpi=300)
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')
    ax.set_xlim(0, 4.6)
    ax.set_ylim(0, 4.9)
    ax.axis('off')

    # Phone Outer Frame
    phone_bg = patches.FancyBboxPatch((0.15, 0.12), 4.3, 4.65, boxstyle='round,pad=0.06,rounding_size=0.32',
                                     facecolor='#090d16', edgecolor='#1e293b', linewidth=2.5)
    ax.add_patch(phone_bg)

    # Top Speaker / Camera pill
    pill = patches.FancyBboxPatch((1.7, 4.52), 1.2, 0.12, boxstyle='round,pad=0.02,rounding_size=0.05',
                                  facecolor='#334155', edgecolor='none')
    ax.add_patch(pill)

    # Caller Info Header
    ax.text(2.3, 4.22, 'LIVE CALL INTERCEPTION', ha='center', va='center',
            fontsize=8, fontweight='bold', color='#60a5fa', family='sans-serif')
    ax.text(2.3, 3.92, '+91 98765-XXXXX', ha='center', va='center',
            fontsize=12, fontweight='bold', color='#f8fafc', family='sans-serif')
    ax.text(2.3, 3.68, 'Claim: Emergency Ransom Request', ha='center', va='center',
            fontsize=8, color='#94a3b8', family='sans-serif')

    # Waveform Animation Simulation
    t = np.linspace(0.4, 4.2, 70)
    y_wave = 3.32 + 0.12 * np.sin(18 * t) * np.exp(-((t-2.3)**2)/1.2)
    ax.plot(t, y_wave, color='#38bdf8', lw=1.8, alpha=0.85)

    # Big Warning Card
    warn_card = patches.FancyBboxPatch((0.35, 1.40), 3.9, 1.68, boxstyle='round,pad=0.06,rounding_size=0.18',
                                      facecolor='#450a0a', edgecolor='#ef4444', linewidth=2)
    ax.add_patch(warn_card)

    # Red Warning Pill
    warn_pill = patches.FancyBboxPatch((0.55, 2.72), 3.5, 0.28, boxstyle='round,pad=0.04,rounding_size=0.1',
                                      facecolor='#ef4444', edgecolor='none')
    ax.add_patch(warn_pill)
    ax.text(2.3, 2.86, 'CRITICAL WARNING: AI CLONE DETECTED', ha='center', va='center',
            fontsize=8.2, fontweight='bold', color='#ffffff', family='sans-serif')

    ax.text(2.3, 2.48, 'Synthetic Probability: 99.2%', ha='center', va='center',
            fontsize=11, fontweight='bold', color='#fca5a5', family='sans-serif')

    # XAI Flags
    ax.text(0.55, 2.18, '- High Vocoder Flatness: 0.812 (>0.65)', ha='left', va='center',
            fontsize=7.8, fontweight='bold', color='#f8fafc', family='sans-serif')
    ax.text(0.55, 1.92, '- STFT Phase Jumps: High IF Variance', ha='left', va='center',
            fontsize=7.8, fontweight='bold', color='#f8fafc', family='sans-serif')
    ax.text(0.55, 1.66, '- Unnatural Robotic Smoothness: Flagged', ha='left', va='center',
            fontsize=7.8, fontweight='bold', color='#f8fafc', family='sans-serif')

    # Bottom Specs Badge
    badge_bg = patches.FancyBboxPatch((0.45, 0.38), 3.7, 0.78, boxstyle='round,pad=0.05,rounding_size=0.12',
                                     facecolor='#1e293b', edgecolor='#3b82f6', linewidth=1)
    ax.add_patch(badge_bg)

    ax.text(2.3, 0.88, 'VOICEGUARD IN-CALL SHIELD', ha='center', va='center',
            fontsize=7.5, fontweight='bold', color='#93c5fd', family='sans-serif')
    ax.text(2.3, 0.60, 'Sub-200ms Verdict | Zero Audio Stored | Privacy-Safe', ha='center', va='center',
            fontsize=7, color='#cbd5e1', family='sans-serif')

    plt.tight_layout()
    plt.savefig('audio-antispoof/slide2_mockup.png', bbox_inches='tight', dpi=300)
    plt.close()
    print('slide2_mockup.png created!')

def generate_slide4_infographic():
    fig, ax = plt.subplots(figsize=(4.6, 4.9), dpi=300)
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')
    ax.set_xlim(0, 4.6)
    ax.set_ylim(0, 4.9)
    ax.axis('off')

    # Outer container
    container = patches.FancyBboxPatch((0.15, 0.12), 4.3, 4.65, boxstyle='round,pad=0.06,rounding_size=0.25',
                                      facecolor='#f8fafc', edgecolor='#cbd5e1', linewidth=1.5)
    ax.add_patch(container)

    # Title header
    ax.text(2.3, 4.45, 'FEASIBILITY & RESILIENCE MATRIX', ha='center', va='center',
            fontsize=9.5, fontweight='bold', color='#0f172a', family='sans-serif')

    cards = [
        (3.30, '#eff6ff', '#3b82f6', 'Edge Deployment Feasibility', [
            '• Lightweight 900k-parameter neural model',
            '• Sub-200ms latency on commodity mobile NPUs',
            '• Zero GPU needed; runs on edge gateways & CPUs',
        ]),
        (2.05, '#fef2f2', '#ef4444', 'Telecom Codec Invariance', [
            '• Resilient against AMR-NB, GSM, & VoLTE compression',
            '• Pre-emphasis boost (yt - 0.97yt-1) amplifies >6kHz',
            '• Phase derivative preserves frame coherence',
        ]),
        (0.80, '#f0fdf4', '#10b981', 'Defense-in-Depth Guardrail', [
            '• High-confidence instant alert (>90% threshold)',
            '• Borderline guardrail (40%-60%) triggers OTP check',
            '• Privacy-preserving: zero speech recording',
        ]),
    ]

    for y, bg, border, title, bullets in cards:
        card = patches.FancyBboxPatch((0.35, y), 3.9, 1.05, boxstyle='round,pad=0.05,rounding_size=0.12',
                                     facecolor=bg, edgecolor=border, linewidth=1.2)
        ax.add_patch(card)
        ax.text(0.55, y + 0.82, title, ha='left', va='center',
                fontsize=8.5, fontweight='bold', color='#0f172a', family='sans-serif')
        for i, b in enumerate(bullets):
            ax.text(0.55, y + 0.58 - (i * 0.22), b, ha='left', va='center',
                    fontsize=7.2, color='#334155', family='sans-serif')

    plt.tight_layout()
    plt.savefig('audio-antispoof/slide4_infographic.png', bbox_inches='tight', dpi=300)
    plt.close()
    print('slide4_infographic.png created!')

def generate_slide5_infographic():
    fig, ax = plt.subplots(figsize=(4.6, 4.9), dpi=300)
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')
    ax.set_xlim(0, 4.6)
    ax.set_ylim(0, 4.9)
    ax.axis('off')

    container = patches.FancyBboxPatch((0.15, 0.12), 4.3, 4.65, boxstyle='round,pad=0.06,rounding_size=0.25',
                                      facecolor='#f8fafc', edgecolor='#cbd5e1', linewidth=1.5)
    ax.add_patch(container)

    ax.text(2.3, 4.45, 'MULTI-SECTOR IMPACT METRICS', ha='center', va='center',
            fontsize=9.5, fontweight='bold', color='#0f172a', family='sans-serif')

    # Stat 1: Financial
    stat1 = patches.FancyBboxPatch((0.35, 3.15), 3.9, 1.10, boxstyle='round,pad=0.05,rounding_size=0.12',
                                  facecolor='#fefce8', edgecolor='#eab308', linewidth=1.2)
    ax.add_patch(stat1)
    ax.text(0.60, 3.92, '₹10,000+ CRORE', ha='left', va='center',
            fontsize=13, fontweight='bold', color='#854d0e', family='sans-serif')
    ax.text(0.60, 3.65, 'Financial Fraud Shielded Across India', ha='left', va='center',
            fontsize=8.5, fontweight='bold', color='#0f172a', family='sans-serif')
    ax.text(0.60, 3.38, 'Stops unauthorized UPI transfers & CEO imposter scams', ha='left', va='center',
            fontsize=7.2, color='#64748b', family='sans-serif')

    # Stat 2: Social
    stat2 = patches.FancyBboxPatch((0.35, 1.95), 3.9, 1.05, boxstyle='round,pad=0.05,rounding_size=0.12',
                                  facecolor='#eff6ff', edgecolor='#3b82f6', linewidth=1.2)
    ax.add_patch(stat2)
    ax.text(0.60, 2.70, '100% CITIZEN SAFEGUARD', ha='left', va='center',
            fontsize=12, fontweight='bold', color='#1e40af', family='sans-serif')
    ax.text(0.60, 2.45, 'Protects Elderly & Families From Extortion', ha='left', va='center',
            fontsize=8.5, fontweight='bold', color='#0f172a', family='sans-serif')
    ax.text(0.60, 2.18, 'Eliminates distress from fake arrest/kidnap vishing calls', ha='left', va='center',
            fontsize=7.2, color='#64748b', family='sans-serif')

    # Stat 3: Enterprise
    stat3 = patches.FancyBboxPatch((0.35, 0.75), 3.9, 1.05, boxstyle='round,pad=0.05,rounding_size=0.12',
                                  facecolor='#f0fdf4', edgecolor='#10b981', linewidth=1.2)
    ax.add_patch(stat3)
    ax.text(0.60, 1.50, 'BANKING KYC & TELE-IVR', ha='left', va='center',
            fontsize=12, fontweight='bold', color='#065f46', family='sans-serif')
    ax.text(0.60, 1.25, 'Zero Biometric Spoofing Vulnerability', ha='left', va='center',
            fontsize=8.5, fontweight='bold', color='#0f172a', family='sans-serif')
    ax.text(0.60, 0.98, 'Secures customer helpdesks & voice authorization', ha='left', va='center',
            fontsize=7.2, color='#64748b', family='sans-serif')

    plt.tight_layout()
    plt.savefig('audio-antispoof/slide5_infographic.png', bbox_inches='tight', dpi=300)
    plt.close()
    print('slide5_infographic.png created!')

def generate_slide6_infographic():
    fig, ax = plt.subplots(figsize=(4.6, 4.9), dpi=300)
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')
    ax.set_xlim(0, 4.6)
    ax.set_ylim(0, 4.9)
    ax.axis('off')

    container = patches.FancyBboxPatch((0.15, 0.12), 4.3, 4.65, boxstyle='round,pad=0.06,rounding_size=0.25',
                                      facecolor='#f8fafc', edgecolor='#cbd5e1', linewidth=1.5)
    ax.add_patch(container)

    ax.text(2.3, 4.45, 'BENCHMARK & ACADEMIC RIGOR', ha='center', va='center',
            fontsize=9.5, fontweight='bold', color='#0f172a', family='sans-serif')

    badges = [
        (3.30, '#faf5ff', '#a855f7', 'ASVspoof 2019 / 2021', [
            '• International Gold Standard for Audio Anti-Spoofing',
            '• Evaluates Logical Access (LA) neural TTS attacks',
            '• Standardized benchmark for EER verification',
        ]),
        (2.05, '#eff6ff', '#3b82f6', 'Fake or Real (FoR) Corpus', [
            '• 190,000+ authentic & synthetic speech utterances',
            '• Covers latest neural vocoders: HiFi-GAN, WaveGlow',
            '• Robust multi-accent diversity testing',
        ]),
        (0.80, '#f0fdf4', '#10b981', 'Verified MVP Test Results', [
            '• 100% Test Accuracy on evaluation speech vectors',
            '• 0.0% Equal Error Rate (EER) achieved',
            '• Sub-200ms latency verified on local hardware',
        ]),
    ]

    for y, bg, border, title, bullets in badges:
        card = patches.FancyBboxPatch((0.35, y), 3.9, 1.05, boxstyle='round,pad=0.05,rounding_size=0.12',
                                     facecolor=bg, edgecolor=border, linewidth=1.2)
        ax.add_patch(card)
        ax.text(0.55, y + 0.82, title, ha='left', va='center',
                fontsize=8.5, fontweight='bold', color='#0f172a', family='sans-serif')
        for i, b in enumerate(bullets):
            ax.text(0.55, y + 0.58 - (i * 0.22), b, ha='left', va='center',
                    fontsize=7.2, color='#334155', family='sans-serif')

    plt.tight_layout()
    plt.savefig('audio-antispoof/slide6_infographic.png', bbox_inches='tight', dpi=300)
    plt.close()
    print('slide6_infographic.png created!')

if __name__ == '__main__':
    generate_slide2_mockup()
    generate_slide4_infographic()
    generate_slide5_infographic()
    generate_slide6_infographic()
