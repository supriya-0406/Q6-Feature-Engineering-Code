
 Q6: `README.md` (Feature Engineering)
```markdown
#  Tophawks Q6: Feature Engineering Pipeline

Transforms raw behavioral/CRM logs into 8 high-signal features for lead conversion prediction. Demonstrates why **commercial intent signals** outperform vanity metrics.

 Features Engineered
1. `pricing_intent` → Time on pricing page, recency-weighted
2. `demo_requested` → Explicit commercial intent
3. `company_size_tier` → Firmographic proxy
4. `recency_score` → Exponential decay of last activity
5. `tech_fit` → Product alignment signal
6. `source_quality` → Acquisition channel tier
7. `domain_is_corporate` → B2B qualification
8. `content_depth` → Research stage ordinal mapping

## 🚀 How to Run
```bash
pip install pandas numpy
python q6_features.py
