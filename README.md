
---

## ⚙️ How It Works
1. Resume and job description text are cleaned and normalized
2. TF-IDF vectorization converts text into numerical features
3. Cosine similarity calculates match percentage
4. Missing keywords are extracted to highlight skill gaps

---

## ▶️ Usage Example
```python
from matcher import match_resume

resume_text = "Python developer with machine learning experience"
job_description = "Looking for a machine learning engineer skilled in Python and NLP"

score, missing_skills = match_resume(resume_text, job_description)

print("Match Score:", score)
print("Missing Skills:", missing_skills)
