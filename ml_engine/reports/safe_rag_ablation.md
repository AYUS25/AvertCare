# Leakage-safe RAG ablation

Neighborhood rates were computed from training encounters only.
Training queries excluded every encounter of the same patient.
Validation and test patients were never in the index.
Notes contain discharge-time fields only.

## Neighborhood-size probe (validation AUROC, one boosted tree)

- tabular: 0.6833
- k5: 0.6797
- k10: 0.6826
- k20: 0.6792

Feature set used for the full grid: **k=10**, plus a tabular-only grid for the comparison.

Winner rule: highest validation AUROC, then validation AUPRC. Test metrics are confirmatory.

| Model | Features | Val AUROC | Val AUPRC | Test AUROC | Test AUPRC | Test F1 | Test Precision | Test Recall | Threshold |
|---|---|---|---|---|---|---|---|---|---|
| LogisticRegression_C0.2 | tabular | 0.6690 | 0.2189 | 0.6681 | 0.2331 | 0.2750 | 0.1905 | 0.4947 | 0.52 |
| LogisticRegression_C1.0 | tabular | 0.6686 | 0.2189 | 0.6680 | 0.2330 | 0.2760 | 0.1912 | 0.4965 | 0.52 |
| LogisticRegression_C5.0 | tabular | 0.6685 | 0.2190 | 0.6680 | 0.2330 | 0.2761 | 0.1912 | 0.4965 | 0.52 |
| RandomForest_d16 | tabular | 0.6799 | 0.2245 | 0.6724 | 0.2314 | 0.2856 | 0.1974 | 0.5159 | 0.45 |
| RandomForest_d24 | tabular | 0.6790 | 0.2231 | 0.6647 | 0.2192 | 0.2757 | 0.1886 | 0.5124 | 0.31 |
| HistGradientBoosting_d6_l31 | tabular | 0.6838 | 0.2336 | 0.6716 | 0.2494 | 0.2929 | 0.2233 | 0.4258 | 0.56 |
| HistGradientBoosting_dNone_l63 | tabular | 0.6792 | 0.2293 | 0.6707 | 0.2488 | 0.2901 | 0.2051 | 0.4956 | 0.52 |
| XGBoost_d4_spw | tabular | 0.6809 | 0.2296 | 0.6737 | 0.2407 | 0.2790 | 0.1899 | 0.5256 | 0.52 |
| XGBoost_d6_spw | tabular | 0.6833 | 0.2321 | 0.6755 | 0.2506 | 0.2851 | 0.2076 | 0.4549 | 0.54 |
| XGBoost_d6_unweighted | tabular | 0.6850 | 0.2384 | 0.6809 | 0.2621 | 0.2868 | 0.2009 | 0.5009 | 0.13 |
| XGBoost_d8_spw | tabular | 0.6844 | 0.2375 | 0.6715 | 0.2461 | 0.2812 | 0.1964 | 0.4947 | 0.51 |
| LogisticRegression_C0.2 | tabular+rag_k10 | 0.6698 | 0.2199 | 0.6677 | 0.2327 | 0.2700 | 0.1708 | 0.6440 | 0.47 |
| LogisticRegression_C1.0 | tabular+rag_k10 | 0.6697 | 0.2201 | 0.6675 | 0.2326 | 0.2758 | 0.1908 | 0.4973 | 0.52 |
| LogisticRegression_C5.0 | tabular+rag_k10 | 0.6697 | 0.2201 | 0.6673 | 0.2326 | 0.2722 | 0.1749 | 0.6140 | 0.48 |
| RandomForest_d16 | tabular+rag_k10 | 0.6790 | 0.2246 | 0.6716 | 0.2302 | 0.2868 | 0.2123 | 0.4417 | 0.47 |
| RandomForest_d24 | tabular+rag_k10 | 0.6780 | 0.2244 | 0.6658 | 0.2175 | 0.2774 | 0.1983 | 0.4611 | 0.32 |
| HistGradientBoosting_d6_l31 | tabular+rag_k10 | 0.6812 | 0.2342 | 0.6735 | 0.2504 | 0.2911 | 0.2231 | 0.4187 | 0.56 |
| HistGradientBoosting_dNone_l63 | tabular+rag_k10 | 0.6827 | 0.2336 | 0.6717 | 0.2438 | 0.2920 | 0.2326 | 0.3922 | 0.57 |
| XGBoost_d4_spw | tabular+rag_k10 | 0.6827 | 0.2354 | 0.6758 | 0.2437 | 0.2838 | 0.1985 | 0.4973 | 0.53 |
| XGBoost_d6_spw | tabular+rag_k10 | 0.6833 | 0.2362 | 0.6744 | 0.2509 | 0.2871 | 0.2044 | 0.4823 | 0.53 |
| XGBoost_d6_unweighted | tabular+rag_k10 | 0.6866 | 0.2388 | 0.6804 | 0.2565 | 0.2905 | 0.1953 | 0.5671 | 0.12 |
| XGBoost_d8_spw | tabular+rag_k10 | 0.6852 | 0.2355 | 0.6696 | 0.2531 | 0.2838 | 0.2081 | 0.4461 | 0.53 |

## Winner

- Model: `XGBoost_d6_unweighted`
- Features: tabular+rag_k10
- Validation AUROC: 0.6866
- Test AUROC: 0.6804
- Test AUPRC: 0.2565

The previous 0.6971 result used notes that contained the readmission label and an index that mixed held-out encounters. It is not comparable to this table.

Dashboard bands for this calibrated model: high risk at probability >= 0.20 (about the top 9% of encounters, test precision about 0.29), moderate risk from the F1 threshold 0.12 up to 0.20. The 0.12 cutoff alone flags about 33% of encounters, so it is not used as the high-risk label.
