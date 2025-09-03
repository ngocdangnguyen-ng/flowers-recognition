# Flowers Recognition - Model Evaluation Report

## Summary
- **Date**: 2025-09-03 13:24:12
- **Best Model**: Fine-tuned ResNet50
- **Test Samples**: 432
- **Classes**: 5 (daisy, dandelion, rose, sunflower, tulip)

## Performance Results
- **Test Accuracy**: 0.6157 (61.6%)
- **Test Loss**: 1.0330
- **Macro F1-Score**: 0.6152
- **Improvement over random**: +41.6%

## Per-Class Performance
| Class | Accuracy |
|-------|----------|
| daisy | 0.579 |
| dandelion | 0.686 |
| rose | 0.570 |
| sunflower | 0.740 |
| tulip | 0.515 |

## Key Insights
1. **Best performing class**: sunflower (0.740 accuracy)
2. **Most challenging class**: tulip (0.515 accuracy)
3. **Overall performance**: Fair model performance
4. **Model reliability**: Moderate confidence in correct predictions

## Recommendations
1. **Production readiness**: Needs improvement before production
2. **Data collection**: Collect more tulip samples
3. **Model improvements**: Consider ensemble methods

## Files Generated
- Performance comparison: `C:\Users\nndng\OneDrive\Desktop\flowers-recognition\evaluation/test_comparison.csv`
- Confusion matrix: `C:\Users\nndng\OneDrive\Desktop\flowers-recognition\evaluation/Fine-tuned ResNet50_confusion_matrix.png`
- Classification report: `C:\Users\nndng\OneDrive\Desktop\flowers-recognition\evaluation/classification_report.txt`

---
*Report generated automatically by evaluation pipeline*
