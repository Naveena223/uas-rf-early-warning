"""
UAS RF Early-Warning — Final Dashboard
========================================
Integrates directly with your pipeline.py, alert_engine.py, classifier.py, library.py.

Run from your project root:
    python dashboard.py
    python dashboard.py --log-dir logs/ --port 8050

Or via Docker (add to Dockerfile CMD):
    CMD ["python", "dashboard.py", "--log-dir", "logs/"]
"""

import os
import sys
import json
import glob
import argparse
import logging
from datetime import datetime, timezone
from typing import List, Dict

import dash
from dash import dcc, html, Input, Output, State, dash_table
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Optional: import your library module for live signature counts ────────────
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from library import SignatureLibrary
    LIB_AVAILABLE = True
except ImportError:
    LIB_AVAILABLE = False

logger = logging.getLogger('uas_dashboard')

# ─────────────────────────────────────────────────────────────────────────────
# Theme tokens — matches your project's dark-professional style
# ─────────────────────────────────────────────────────────────────────────────
C = {
    'bg':       '#0B0F1A',
    'card':     '#111827',
    'border':   '#1E2A3A',
    'accent':   '#00D4AA',
    'blue':     '#3B82F6',
    'uas':      '#EF4444',
    'unk':      '#F59E0B',
    'non':      '#10B981',
    'text':     '#E5E7EB',
    'muted':    '#6B7280',
    'high':     '#EF4444',
    'med':      '#F59E0B',
    'low':      '#10B981',
}

CARD = {
    'backgroundColor': C['card'],
    'border': f'1px solid {C["border"]}',
    'borderRadius': '10px',
    'padding': '16px',
    'marginBottom': '12px',
}

LABEL_STYLE = {
    'fontSize': '10px',
    'color': C['muted'],
    'letterSpacing': '1.5px',
    'textTransform': 'uppercase',
    'fontFamily': 'monospace',
    'marginBottom': '6px',
}

# ─────────────────────────────────────────────────────────────────────────────
# Data loaders — reads YOUR alerts_*.json JSONL logs and summary_*.json
# ─────────────────────────────────────────────────────────────────────────────

def load_alerts(log_dir: str) -> List[Dict]:
    alerts = []
    for pattern in [f'{log_dir}/alerts_*.json', f'{log_dir}/alerts.json']:
        for f in sorted(glob.glob(pattern)):
            with open(f) as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            alerts.append(json.loads(line))
                        except Exception:
                            pass
    return sorted(alerts, key=lambda a: a.get('epoch_s', 0))


def load_summaries(log_dir: str) -> List[Dict]:
    summaries = []
    for f in sorted(glob.glob(f'{log_dir}/summary_*.json')):
        try:
            with open(f) as fh:
                summaries.append(json.load(fh))
        except Exception:
            pass
    return summaries


def load_library(library_path: str) -> List[Dict]:
    entries = []
    if not os.path.exists(library_path):
        return entries
    with open(library_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except Exception:
                    pass
    return entries


# ─────────────────────────────────────────────────────────────────────────────
# Plotly chart builders — all use your alert fields exactly
# ─────────────────────────────────────────────────────────────────────────────

DARK_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color=C['text'], family='monospace', size=11),
    margin=dict(l=50, r=20, t=30, b=40),
    xaxis=dict(gridcolor=C['border'], zerolinecolor=C['border'], showgrid=True),
    yaxis=dict(gridcolor=C['border'], zerolinecolor=C['border'], showgrid=True),
    legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor=C['border']),
)


def fig_timeline(alerts: List[Dict], threshold: float = 0.55) -> go.Figure:
    """Confidence vs. time scatter — UAS-LIKE red, UNKNOWN amber."""
    uas  = [a for a in alerts if a['alert_type'] == 'UAS-LIKE']
    unk  = [a for a in alerts if a['alert_type'] == 'UNKNOWN']

    fig = go.Figure()
    if uas:
        fig.add_trace(go.Scatter(
            x=[a['signal_time_start'] for a in uas],
            y=[a['confidence'] for a in uas],
            mode='markers',
            name='UAS-LIKE',
            marker=dict(color=C['uas'], size=6, opacity=0.8),
            hovertemplate='<b>%{text}</b><br>conf=%{y:.3f}<br>t=%{x:.3f}s',
            text=[f"{a['alert_id']} | {a['center_freq_hz']/1e6:.1f}MHz" for a in uas],
        ))
    if unk:
        fig.add_trace(go.Scatter(
            x=[a['signal_time_start'] for a in unk],
            y=[a['confidence'] for a in unk],
            mode='markers',
            name='UNKNOWN',
            marker=dict(color=C['unk'], size=5, opacity=0.7, symbol='diamond'),
            hovertemplate='conf=%{y:.3f}<br>t=%{x:.3f}s',
        ))
    # Threshold line
    fig.add_hline(y=threshold, line_dash='dash',
                  line_color=C['accent'], line_width=1,
                  annotation_text=f'threshold {threshold}',
                  annotation_font_color=C['accent'],
                  annotation_font_size=10)
    layout = dict(**DARK_LAYOUT)
    layout['yaxis'] = dict(range=[0, 1.05], gridcolor=C['border'])
    fig.update_layout(**layout,
                      title=dict(text='Alert Confidence Timeline', font=dict(size=12, color=C['muted'])),
                      xaxis_title='Signal Time (s)',
                      yaxis_title='Confidence')
    return fig


def fig_frequency(alerts: List[Dict]) -> go.Figure:
    """Frequency vs. confidence scatter with UAS band shading."""
    uas  = [a for a in alerts if a['alert_type'] == 'UAS-LIKE']
    unk  = [a for a in alerts if a['alert_type'] == 'UNKNOWN']

    fig = go.Figure()

    # UAS band overlays
    uas_bands = [(2400, 2483, '2.4G ISM'), (5725, 5850, '5.8G ISM'),
                 (902, 928, '900M ISM'), (430, 435, '433M')]
    for lo, hi, name in uas_bands:
        fig.add_vrect(x0=lo, x1=hi, fillcolor=C['accent'],
                      opacity=0.07, line_width=0)
        fig.add_annotation(x=(lo+hi)/2, y=1.03, text=name, yref='paper',
                           font=dict(size=9, color=C['accent']),
                           showarrow=False)

    if uas:
        fig.add_trace(go.Scatter(
            x=[a['center_freq_hz']/1e6 for a in uas],
            y=[a['confidence'] for a in uas],
            mode='markers',
            name='UAS-LIKE',
            marker=dict(color=C['uas'], size=6, opacity=0.75),
            hovertemplate='%{x:.1f} MHz<br>conf=%{y:.3f}',
        ))
    if unk:
        fig.add_trace(go.Scatter(
            x=[a['center_freq_hz']/1e6 for a in unk],
            y=[a['confidence'] for a in unk],
            mode='markers',
            name='UNKNOWN',
            marker=dict(color=C['unk'], size=5, opacity=0.7, symbol='diamond'),
            hovertemplate='%{x:.1f} MHz<br>conf=%{y:.3f}',
        ))
    layout2 = dict(**DARK_LAYOUT)
    layout2['yaxis'] = dict(range=[0, 1.1], gridcolor=C['border'])
    fig.update_layout(**layout2,
                      title=dict(text='Frequency vs. Confidence (green bands = UAS bands)',
                                 font=dict(size=12, color=C['muted'])),
                      xaxis_title='Frequency (MHz)',
                      yaxis_title='Confidence')
    return fig


def fig_latency(alerts: List[Dict]) -> go.Figure:
    """Latency histogram + 2000 ms budget line."""
    lats = [a['detection_latency_ms'] for a in alerts if 'detection_latency_ms' in a]
    if not lats:
        return go.Figure().update_layout(**DARK_LAYOUT)

    fig = go.Figure(go.Histogram(
        x=lats, nbinsx=40,
        marker=dict(color=C['blue'], opacity=0.8,
                    line=dict(color=C['border'], width=0.5)),
        name='Latency (ms)',
    ))
    fig.add_vline(x=2000, line_dash='dash', line_color=C['uas'], line_width=1,
                  annotation_text='2000ms budget', annotation_font_color=C['uas'],
                  annotation_font_size=10)
    fig.update_layout(**DARK_LAYOUT,
                      title=dict(text='Alert Latency Distribution',
                                 font=dict(size=12, color=C['muted'])),
                      xaxis_title='Latency (ms)',
                      yaxis_title='Count',
                      showlegend=False)
    return fig


def fig_severity(alerts: List[Dict]) -> go.Figure:
    """Severity donut chart."""
    sev_counts = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
    for a in alerts:
        sev_counts[a.get('severity', 'LOW')] = sev_counts.get(a.get('severity', 'LOW'), 0) + 1

    fig = go.Figure(go.Pie(
        labels=list(sev_counts.keys()),
        values=list(sev_counts.values()),
        hole=0.60,
        marker=dict(colors=[C['high'], C['med'], C['low']],
                    line=dict(color=C['card'], width=2)),
        textfont=dict(size=11, color=C['text']),
        hovertemplate='%{label}: %{value}<extra></extra>',
    ))
    layout3 = {k: v for k, v in DARK_LAYOUT.items() if k not in ('margin', 'legend')}
    fig.update_layout(**layout3,
                      title=dict(text='Alert Severity', font=dict(size=12, color=C['muted'])),
                      margin=dict(l=20, r=20, t=40, b=20),
                      showlegend=True,
                      legend=dict(orientation='h', y=-0.1, bgcolor='rgba(0,0,0,0)'))
    return fig


def fig_confidence_hist(alerts: List[Dict]) -> go.Figure:
    """Confidence distribution split by type."""
    uas_confs = [a['confidence'] for a in alerts if a['alert_type'] == 'UAS-LIKE']
    unk_confs = [a['confidence'] for a in alerts if a['alert_type'] == 'UNKNOWN']

    fig = go.Figure()
    if uas_confs:
        fig.add_trace(go.Histogram(x=uas_confs, nbinsx=25, name='UAS-LIKE',
                                   marker_color=C['uas'], opacity=0.7))
    if unk_confs:
        fig.add_trace(go.Histogram(x=unk_confs, nbinsx=15, name='UNKNOWN',
                                   marker_color=C['unk'], opacity=0.7))
    fig.update_layout(**DARK_LAYOUT,
                      title=dict(text='Confidence Distribution',
                                 font=dict(size=12, color=C['muted'])),
                      xaxis_title='Confidence Score',
                      yaxis_title='Count',
                      barmode='overlay')
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# KPI helpers
# ─────────────────────────────────────────────────────────────────────────────

def kpi_card(value: str, label: str, color: str,
             sub: str = '', delta: str = '') -> html.Div:
    return html.Div([
        html.Div(value, style={
            'fontSize': '32px', 'fontWeight': '500',
            'color': color, 'lineHeight': '1', 'fontFamily': 'monospace',
        }),
        html.Div(label, style=LABEL_STYLE),
        html.Div(sub, style={'fontSize': '11px', 'color': C['muted']}) if sub else html.Div(),
    ], style={**CARD, 'flex': '1', 'textAlign': 'center', 'marginBottom': '0'})


def badge(text: str, color: str, bg: str) -> html.Span:
    return html.Span(text, style={
        'background': bg, 'color': color,
        'padding': '2px 8px', 'borderRadius': '20px',
        'fontSize': '10px', 'fontWeight': '500',
    })


# ─────────────────────────────────────────────────────────────────────────────
# Alert table rows — uses YOUR exact alert fields
# ─────────────────────────────────────────────────────────────────────────────

def alert_table_rows(alerts: List[Dict], limit: int = 40) -> html.Table:
    recent = sorted(alerts, key=lambda a: a.get('epoch_s', 0), reverse=True)[:limit]

    def type_badge(t):
        if t == 'UAS-LIKE':
            return badge('UAS-LIKE', '#7F1D1D', '#FEE2E2')
        return badge('UNKNOWN',  '#78350F', '#FEF3C7')

    def sev_badge(s):
        cfg = {'HIGH': ('#7F1D1D', '#FEE2E2'),
               'MEDIUM': ('#78350F', '#FEF3C7'),
               'LOW': ('#14532D', '#DCFCE7')}
        c, bg = cfg.get(s, ('#374151', '#F3F4F6'))
        return badge(s, c, bg)

    def lat_color(ms):
        return C['uas'] if ms > 200 else C['accent'] if ms > 50 else C['non']

    rows = []
    for a in recent:
        freq_mhz = a.get('center_freq_hz', 0) / 1e6
        bw_khz   = a.get('bandwidth_hz', 0) / 1e3
        lat      = a.get('detection_latency_ms', 0)
        rows.append(html.Tr([
            html.Td(a.get('alert_id', ''), style={'fontFamily': 'monospace',
                    'fontSize': '10px', 'color': C['accent']}),
            html.Td(type_badge(a.get('alert_type', ''))),
            html.Td(sev_badge(a.get('severity', ''))),
            html.Td(f'{freq_mhz:.2f}', style={'fontFamily': 'monospace', 'fontSize': '11px'}),
            html.Td(f'{bw_khz:.0f}', style={'fontFamily': 'monospace', 'fontSize': '11px'}),
            html.Td(f'{a.get("confidence", 0):.3f}',
                    style={'fontFamily': 'monospace', 'fontSize': '11px',
                           'color': C['uas'] if a.get('confidence', 0) > 0.8 else C['text']}),
            html.Td(f'{lat:.0f}ms',
                    style={'fontFamily': 'monospace', 'fontSize': '11px',
                           'color': lat_color(lat)}),
            html.Td(a.get('source_file', ''),
                    style={'fontSize': '10px', 'color': C['muted'],
                           'maxWidth': '120px', 'overflow': 'hidden',
                           'textOverflow': 'ellipsis', 'whiteSpace': 'nowrap'}),
        ], style={'borderBottom': f'1px solid {C["border"]}'}))

    th_style = {'fontSize': '10px', 'color': C['muted'], 'letterSpacing': '1px',
                'textTransform': 'uppercase', 'padding': '6px 8px',
                'borderBottom': f'2px solid {C["border"]}', 'textAlign': 'left'}

    return html.Table([
        html.Thead(html.Tr([
            html.Th('ID', style=th_style),
            html.Th('Type', style=th_style),
            html.Th('Sev', style=th_style),
            html.Th('Freq MHz', style=th_style),
            html.Th('BW kHz', style=th_style),
            html.Th('Conf', style=th_style),
            html.Th('Latency', style=th_style),
            html.Th('Source', style=th_style),
        ])),
        html.Tbody(rows),
    ], style={'width': '100%', 'borderCollapse': 'collapse',
              'fontSize': '12px', 'fontFamily': 'monospace'})


# ─────────────────────────────────────────────────────────────────────────────
# Evidence snapshot — shows features dict from YOUR alert_engine
# ─────────────────────────────────────────────────────────────────────────────

def evidence_panel(alerts: List[Dict]) -> html.Div:
    uas = [a for a in alerts if a['alert_type'] == 'UAS-LIKE']
    if not uas:
        return html.Div('No UAS-LIKE alerts yet.',
                        style={'color': C['muted'], 'fontSize': '12px'})
    a = uas[-1]
    feats = a.get('features', {})
    lib_id = a.get('library_match_id') or 'new'

    def row(k, v):
        return html.Div([
            html.Span(k, style={'color': C['muted'], 'fontSize': '11px',
                                'minWidth': '160px', 'display': 'inline-block'}),
            html.Span(v, style={'color': C['accent'], 'fontSize': '11px',
                                'fontFamily': 'monospace'}),
        ], style={'padding': '3px 0', 'borderBottom': f'1px solid {C["border"]}'})

    return html.Div([
        html.Div([
            html.Span(a['alert_id'], style={'fontFamily': 'monospace',
                      'fontSize': '14px', 'fontWeight': '500', 'color': C['accent']}),
            badge('UAS-LIKE', '#7F1D1D', '#FEE2E2'),
        ], style={'display': 'flex', 'justifyContent': 'space-between',
                  'alignItems': 'center', 'marginBottom': '10px'}),

        row('Frequency',       f'{a.get("center_freq_hz",0)/1e6:.3f} MHz'),
        row('Bandwidth',       f'{a.get("bandwidth_hz",0)/1e3:.1f} kHz'),
        row('Confidence',      f'{a.get("confidence",0):.4f}'),
        row('Severity',        a.get('severity', '')),
        row('Latency',         f'{a.get("detection_latency_ms",0):.1f} ms'),
        row('Source',          a.get('source_file', '')),
        row('Library match',   lib_id),
        row('Time',            f'{a.get("signal_time_start",0):.3f}s – {a.get("signal_time_end",0):.3f}s'),
        html.Div('Key Features', style={**LABEL_STYLE, 'marginTop': '10px'}),
    ] + [row(k, f'{v:.4f}') for k, v in list(feats.items())[:6]])


# ─────────────────────────────────────────────────────────────────────────────
# Library panel — reads your library/uas_signatures.jsonl
# ─────────────────────────────────────────────────────────────────────────────

def library_panel(entries: List[Dict]) -> html.Div:
    if not entries:
        return html.Div('No signatures stored yet.',
                        style={'color': C['muted'], 'fontSize': '12px'})

    def entry_card(e):
        occ = e.get('occurrence_count', 1)
        return html.Div([
            html.Div([
                html.Span(e.get('signature_id', ''),
                          style={'fontFamily': 'monospace', 'fontSize': '11px',
                                 'color': C['blue'], 'fontWeight': '500'}),
                html.Span(f'×{occ}',
                          style={'fontSize': '11px', 'color': C['accent'],
                                 'fontWeight': '500'}),
            ], style={'display': 'flex', 'justifyContent': 'space-between'}),
            html.Div(e.get('summary', ''),
                     style={'fontSize': '10px', 'color': C['muted'],
                            'marginTop': '3px', 'lineHeight': '1.4'}),
        ], style={'padding': '8px 0', 'borderBottom': f'1px solid {C["border"]}'})

    return html.Div([entry_card(e) for e in entries[-10:]])


# ─────────────────────────────────────────────────────────────────────────────
# Performance metrics panel — reads your summary_*.json
# ─────────────────────────────────────────────────────────────────────────────

def perf_panel(summaries: List[Dict]) -> html.Div:
    if not summaries:
        return html.Div('No summary data.', style={'color': C['muted'], 'fontSize': '12px'})

    s = summaries[-1]
    perf = s.get('performance', {})
    als  = s.get('alert_stats', {})
    ds   = s.get('detection_stats', {})
    ri   = s.get('run_info', {})

    def row(label, val, ok=None):
        color = C['non'] if ok is True else C['uas'] if ok is False else C['accent']
        return html.Div([
            html.Span(label, style={'color': C['muted'], 'fontSize': '11px',
                                    'minWidth': '200px', 'display': 'inline-block'}),
            html.Span(str(val), style={'color': color, 'fontSize': '11px',
                                       'fontFamily': 'monospace', 'fontWeight': '500'}),
        ], style={'padding': '4px 0', 'borderBottom': f'1px solid {C["border"]}'})

    lat_max = als.get('latency_max_ms', 0)
    return html.Div([
        row('Segments processed',   ds.get('segments_processed', 0)),
        row('UAS-like detections',  ds.get('uas_detections', 0)),
        row('Unknown detections',   ds.get('unknown_detections', 0)),
        row('Non-UAS detections',   ds.get('non_uas_detections', 0)),
        row('Mean segment time',    f'{perf.get("mean_segment_time_ms",0):.1f} ms'),
        row('Throughput',           f'{perf.get("throughput_segs_per_s",0):.1f} segs/s'),
        row('Alert latency mean',   f'{als.get("latency_mean_ms",0):.1f} ms'),
        row('Alert latency max',    f'{lat_max:.1f} ms',
            ok=lat_max <= 2000),
        row('Budget met (<2000ms)', f'{als.get("latency_budget_met_pct",100):.1f}%',
            ok=als.get('latency_budget_met_pct', 100) == 100.0),
        row('Library signatures',   ri.get('library_size', 0)),
        row('Run duration',         f'{ri.get("run_duration_s",0):.2f}s'),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Full Dash layout
# ─────────────────────────────────────────────────────────────────────────────

def build_app(log_dir: str, library_path: str, refresh_ms: int = 5000) -> dash.Dash:
    app = dash.Dash(__name__, title='UAS RF Early-Warning')

    app.layout = html.Div([

        # ── Header ──────────────────────────────────────────────────────────
        html.Div([
            html.Div([
                html.Div('UAS RF EARLY-WARNING SYSTEM', style={
                    'fontSize': '14px', 'fontWeight': '500',
                    'color': C['accent'], 'letterSpacing': '2px',
                    'fontFamily': 'monospace',
                }),
                html.Div('PASSIVE MONITORING · OFFLINE · DEFENSIVE ONLY', style={
                    'fontSize': '10px', 'color': C['muted'],
                    'letterSpacing': '1.5px', 'fontFamily': 'monospace',
                }),
            ]),
            html.Div([
                html.Span('● ACTIVE', style={
                    'color': C['accent'], 'fontSize': '11px',
                    'fontFamily': 'monospace', 'fontWeight': '500',
                    'animation': 'none',
                }),
                html.Div(id='last-update',
                         style={'fontSize': '10px', 'color': C['muted'],
                                'fontFamily': 'monospace', 'marginTop': '2px'}),
            ], style={'textAlign': 'right'}),
        ], style={
            'backgroundColor': '#05080F',
            'borderBottom': f'2px solid {C["accent"]}',
            'padding': '14px 24px',
            'display': 'flex',
            'justifyContent': 'space-between',
            'alignItems': 'center',
        }),

        # ── Controls bar ─────────────────────────────────────────────────────
        html.Div([
            html.Span('Filter type:', style=LABEL_STYLE),
            dcc.Dropdown(
                id='filter-type',
                options=[
                    {'label': 'All alerts', 'value': 'all'},
                    {'label': 'UAS-LIKE only', 'value': 'UAS-LIKE'},
                    {'label': 'UNKNOWN only', 'value': 'UNKNOWN'},
                ],
                value='all',
                clearable=False,
                style={'width': '160px', 'fontSize': '12px'},
            ),
            html.Span('Severity:', style={**LABEL_STYLE, 'marginLeft': '12px'}),
            dcc.Dropdown(
                id='filter-sev',
                options=[
                    {'label': 'All', 'value': 'all'},
                    {'label': 'HIGH', 'value': 'HIGH'},
                    {'label': 'MEDIUM', 'value': 'MEDIUM'},
                    {'label': 'LOW', 'value': 'LOW'},
                ],
                value='all',
                clearable=False,
                style={'width': '130px', 'fontSize': '12px'},
            ),
            html.Span('Threshold:', style={**LABEL_STYLE, 'marginLeft': '16px'}),
            dcc.Slider(id='threshold-slider', min=0.1, max=0.9, step=0.05,
                       value=0.55, marks={0.1: '0.1', 0.55: '0.55', 0.9: '0.9'},
                       tooltip={'always_visible': True, 'placement': 'top'},
                       className=''),
            html.Span(id='thr-label', style={
                'fontSize': '12px', 'color': C['accent'],
                'fontFamily': 'monospace', 'marginLeft': '6px',
            }),
        ], style={
            'backgroundColor': '#0D1220',
            'padding': '10px 24px',
            'display': 'flex',
            'alignItems': 'center',
            'gap': '8px',
            'borderBottom': f'1px solid {C["border"]}',
            'flexWrap': 'wrap',
        }),

        # ── KPI Row ──────────────────────────────────────────────────────────
        html.Div(id='kpi-row', style={
            'display': 'flex', 'gap': '12px',
            'padding': '16px 24px 0',
        }),

        # ── Main content ─────────────────────────────────────────────────────
        html.Div([
            # Left column: charts
            html.Div([
                html.Div([
                    html.Div('Alert Timeline', style=LABEL_STYLE),
                    dcc.Graph(id='timeline-chart',
                              style={'height': '210px'},
                              config={'displayModeBar': False}),
                ], style=CARD),

                html.Div([
                    html.Div('Frequency Distribution', style=LABEL_STYLE),
                    dcc.Graph(id='freq-chart',
                              style={'height': '190px'},
                              config={'displayModeBar': False}),
                ], style=CARD),

                html.Div([
                    html.Div(style={'display': 'flex', 'gap': '12px'}, children=[
                        html.Div([
                            html.Div('Latency Distribution', style=LABEL_STYLE),
                            dcc.Graph(id='latency-chart',
                                      style={'height': '160px'},
                                      config={'displayModeBar': False}),
                        ], style={'flex': '1'}),
                        html.Div([
                            html.Div('Severity Split', style=LABEL_STYLE),
                            dcc.Graph(id='severity-chart',
                                      style={'height': '160px'},
                                      config={'displayModeBar': False}),
                        ], style={'flex': '1'}),
                    ]),
                ], style=CARD),

            ], style={'flex': '2', 'minWidth': '0'}),

            # Right column: table + evidence + library + perf
            html.Div([
                html.Div([
                    html.Div('Recent Alerts', style=LABEL_STYLE),
                    html.Div(id='alert-table', style={
                        'overflowX': 'auto', 'overflowY': 'auto',
                        'maxHeight': '260px',
                    }),
                ], style=CARD),

                html.Div([
                    html.Div('Evidence Snapshot — Latest UAS-LIKE', style=LABEL_STYLE),
                    html.Div(id='evidence-panel'),
                ], style=CARD),

                html.Div([
                    html.Div('Signature Library', style=LABEL_STYLE),
                    html.Div(id='library-panel'),
                ], style=CARD),

                html.Div([
                    html.Div('Pipeline Performance', style=LABEL_STYLE),
                    html.Div(id='perf-panel'),
                ], style=CARD),

            ], style={'flex': '1.3', 'minWidth': '0'}),

        ], style={
            'display': 'flex', 'gap': '12px',
            'padding': '12px 24px 24px',
            'alignItems': 'flex-start',
        }),

        # Auto-refresh interval + hidden stores
        dcc.Interval(id='interval', interval=refresh_ms, n_intervals=0),
        dcc.Store(id='log-dir-store', data=log_dir),
        dcc.Store(id='lib-path-store', data=library_path),

    ], style={
        'backgroundColor': C['bg'],
        'minHeight': '100vh',
        'color': C['text'],
        'fontFamily': 'system-ui, monospace',
    })

    # ── Callbacks ─────────────────────────────────────────────────────────────

    @app.callback(
        Output('last-update', 'children'),
        Output('kpi-row', 'children'),
        Output('timeline-chart', 'figure'),
        Output('freq-chart', 'figure'),
        Output('latency-chart', 'figure'),
        Output('severity-chart', 'figure'),
        Output('alert-table', 'children'),
        Output('evidence-panel', 'children'),
        Output('library-panel', 'children'),
        Output('perf-panel', 'children'),
        Input('interval', 'n_intervals'),
        Input('filter-type', 'value'),
        Input('filter-sev', 'value'),
        Input('threshold-slider', 'value'),
        State('log-dir-store', 'data'),
        State('lib-path-store', 'data'),
    )
    def refresh(n, f_type, f_sev, threshold, ld, lp):
        # Load fresh data every tick
        all_alerts  = load_alerts(ld)
        summaries   = load_summaries(ld)
        lib_entries = load_library(lp)

        # Apply filters
        alerts = [a for a in all_alerts
                  if (f_type == 'all' or a.get('alert_type') == f_type)
                  and (f_sev == 'all'  or a.get('severity')   == f_sev)
                  and a.get('confidence', 0) >= threshold - 0.10]

        uas   = [a for a in alerts if a.get('alert_type') == 'UAS-LIKE']
        unk   = [a for a in alerts if a.get('alert_type') == 'UNKNOWN']
        lats  = [a.get('detection_latency_ms', 0) for a in alerts]

        # KPIs
        mean_conf = sum(a.get('confidence', 0) for a in alerts) / len(alerts) if alerts else 0
        max_lat   = max(lats) if lats else 0

        kpis = html.Div([
            kpi_card(str(len(uas)),   'UAS-Like Alerts', C['uas']),
            kpi_card(str(len(unk)),   'Unknown',         C['unk']),
            kpi_card(f'{mean_conf:.2f}', 'Mean Confidence', C['accent']),
            kpi_card(f'{max_lat:.0f}ms', 'Max Latency',
                     C['non'] if max_lat <= 2000 else C['uas'],
                     sub='req ≤ 2000ms'),
            kpi_card(str(len(lib_entries)), 'Library Signatures', C['blue']),
        ], style={'display': 'flex', 'gap': '12px', 'width': '100%'})

        now = datetime.now(timezone.utc).strftime('%H:%M:%S UTC')

        return (
            f'Last refresh: {now}',
            kpis,
            fig_timeline(alerts, threshold),
            fig_frequency(alerts),
            fig_latency(alerts),
            fig_severity(alerts),
            alert_table_rows(alerts),
            evidence_panel(alerts),
            library_panel(lib_entries),
            perf_panel(summaries),
        )

    @app.callback(
        Output('thr-label', 'children'),
        Input('threshold-slider', 'value'),
    )
    def update_thr_label(v):
        return f'{v:.2f}'

    return app


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s')

    parser = argparse.ArgumentParser(description='UAS RF Early-Warning Dashboard')
    parser.add_argument('--log-dir',      default='logs',
                        help='Directory containing alerts_*.json and summary_*.json')
    parser.add_argument('--library-path', default='library/uas_signatures.jsonl',
                        help='Path to UAS signature library JSONL file')
    parser.add_argument('--port',         type=int, default=8050)
    parser.add_argument('--refresh',      type=int, default=5000,
                        help='Auto-refresh interval in milliseconds')
    parser.add_argument('--debug',        action='store_true')
    args = parser.parse_args()

    print(f"""
╔═══════════════════════════════════════════════════╗
║   UAS RF Early-Warning Dashboard                  ║
║   http://localhost:{args.port:<5}                        ║
║   Log dir  : {args.log_dir:<36}║
║   Library  : {args.library_path:<36}║
║   Refresh  : {args.refresh}ms                              ║
╚═══════════════════════════════════════════════════╝
    """)

    app = build_app(
        log_dir=args.log_dir,
        library_path=args.library_path,
        refresh_ms=args.refresh,
    )
    app.run(debug=args.debug, port=args.port, host='0.0.0.0')


if __name__ == '__main__':
    main()
