import pandas as pd
from app import _load_data, to_json, fmt_gbp, _INTELLIGENCE, SEGMENT_COLORS
import plotly.graph_objects as go

df_c, df_r, model = _load_data()
df = df_c

try:
    cid = df['CustomerID'].iloc[0]
    print('Testing CustomerID:', cid)
    row_df = df[df['CustomerID'] == cid]
    r = row_df.iloc[0]
    seg = r['Segment']
    clr = SEGMENT_COLORS.get(seg, '#6366f1')

    r_pct = 1.0 - (r['Recency']  / df['Recency'].max())
    f_pct =        r['Frequency'] / df['Frequency'].max()
    m_pct =        r['Monetary']  / df['Monetary'].max()
    pct_rank = (df['Monetary'] < r['Monetary']).mean()

    cats = ['Recency', 'Frequency', 'Monetary', 'Recency']
    vals = [round(r_pct, 3), round(f_pct, 3), round(m_pct, 3), round(r_pct, 3)]
    fig_radar = go.Figure(go.Scatterpolar(
        r=vals, theta=cats, fill='toself',
        fillcolor=f'{clr}1a',
        line=dict(color=clr, width=2.5),
        marker=dict(size=8, color=clr),
        hovertemplate='%{theta}: <b>%{r:.0%}</b><extra></extra>',
    ))

    customer = {
        'id':         cid,
        'segment':    seg,
        'color':      clr,
        'icon':       {'VIP': 'bi-gem', 'Regular': 'bi-cart3',
                       'At Risk': 'bi-exclamation-triangle'}.get(seg, 'bi-person'),
        'seg_class':  {'VIP': 'seg-vip', 'Regular': 'seg-reg',
                       'At Risk': 'seg-risk'}.get(seg, 'seg-reg'),
        'recency':    int(r['Recency']),
        'frequency':  int(r['Frequency']),
        'monetary':   fmt_gbp(r['Monetary']),
        'r_pct':      round(r_pct * 100, 1),
        'f_pct':      round(f_pct * 100, 1),
        'm_pct':      round(m_pct * 100, 1),
        'top_pct':    f'{100 - pct_rank*100:.0f}',
        'avg_recency':  f'{df[\"Recency\"].mean():.0f}',
        'avg_frequency': f'{df[\"Frequency\"].mean():.1f}',
        'avg_monetary':  fmt_gbp(df['Monetary'].mean()),
        'chart_radar':   to_json(fig_radar),
        'intel':         _INTELLIGENCE.get(seg, _INTELLIGENCE['Regular']),
    }
    print('SUCCESS')
except Exception as e:
    import traceback
    traceback.print_exc()
