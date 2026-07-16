# Phase 5.1: Manipulation Check Report

**Generated**: 2026-07-12T18:30:06.198400
**Model**: Qwen3-32B-Q5_K_M
**Samples per condition**: 5

## MP2

### Option Calibration
| Option | Mean Probability |
|--------|------------------|
| A | 29.7% |
| B | 11.7% |
| C | 25.8% |
| D | 32.8% |

**Result**: ✅ PASS
**Reason**: No option exceeded 55% average probability. Max = 32.8%.

### Gap Check
#### Gap: HIGH
- **Mean Entropy**: 1.91 ± 0.06
- **Effective Choices**: 3.2
- **Confidence**: 65.6% ± 12.8%

#### Gap: LOW
- **Mean Entropy**: 1.96 ± 0.04
- **Effective Choices**: 3.6
- **Confidence**: 58.0% ± 10.3%

**Gap Check Result**: ❌ FAIL
  - Low Gap effective choices (3.6) > 2
  - High entropy (1.91) <= Low entropy (1.96)

### Match Check
#### Gap: HIGH
**Correct Goal**
- Top-1 Change Rate: 2/5 (40%)
- Reason Consistency: Strong 0/5 (0%), Weak 0/5 (0%), Unsupported 5/5 (100%)
**Wrong Goal**
- Top-1 Change Rate: 4/5 (80%)
- Reason Consistency: Strong 0/5 (0%), Weak 0/5 (0%), Unsupported 5/5 (100%)

#### Gap: LOW
**Correct Goal**
- Top-1 Change Rate: 1/5 (20%)
- Reason Consistency: Strong 0/5 (0%), Weak 0/5 (0%), Unsupported 5/5 (100%)
**Wrong Goal**
- Top-1 Change Rate: 3/5 (60%)
- Reason Consistency: Strong 0/5 (0%), Weak 0/5 (0%), Unsupported 5/5 (100%)

**Match Check Result**: ❌ FAIL
  - high Correct Goal: 40% < 70%
  - high Wrong Goal: 80% > 40%
  - low Correct Goal: 20% < 70%
  - low Wrong Goal: 60% > 40%

## MP3

### Option Calibration
| Option | Mean Probability |
|--------|------------------|
| A | 33.0% |
| B | 17.0% |
| C | 28.0% |
| D | 22.0% |

**Result**: ✅ PASS
**Reason**: No option exceeded 55% average probability. Max = 33.0%.

### Gap Check
#### Gap: HIGH
- **Mean Entropy**: 1.97 ± 0.04
- **Effective Choices**: 3.8
- **Confidence**: 50.0% ± 0.0%

#### Gap: LOW
- **Mean Entropy**: 1.99 ± 0.03
- **Effective Choices**: 3.8
- **Confidence**: 50.0% ± 0.0%

**Gap Check Result**: ❌ FAIL
  - Low Gap effective choices (3.8) > 2
  - High entropy (1.97) <= Low entropy (1.99)

### Match Check
#### Gap: HIGH
**Correct Goal**
- Top-1 Change Rate: 2/5 (40%)
- Reason Consistency: Strong 0/5 (0%), Weak 0/5 (0%), Unsupported 5/5 (100%)
**Wrong Goal**
- Top-1 Change Rate: 2/5 (40%)
- Reason Consistency: Strong 0/5 (0%), Weak 0/5 (0%), Unsupported 5/5 (100%)

#### Gap: LOW
**Correct Goal**
- Top-1 Change Rate: 0/5 (0%)
- Reason Consistency: Strong 0/5 (0%), Weak 0/5 (0%), Unsupported 5/5 (100%)
**Wrong Goal**
- Top-1 Change Rate: 0/5 (0%)
- Reason Consistency: Strong 0/5 (0%), Weak 0/5 (0%), Unsupported 5/5 (100%)

**Match Check Result**: ❌ FAIL
  - high Correct Goal: 40% < 70%
  - low Correct Goal: 0% < 70%

---

## Overall Summary

| Check | MP2 | MP3 |
|-------|-----|-----|
| MP2 | ❌ FAIL |
| MP3 | ❌ FAIL |

### Diagnosis
- MP2: FAIL
- MP3: FAIL

**Overall Decision: NO-GO**
Manipulation failed for: MP2, MP3
Revise experiment materials before Phase 5.2.