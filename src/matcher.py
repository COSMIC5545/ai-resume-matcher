from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def match_resume(resume_text, job_description):
    documents = [resume_text, job_description]

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(documents)

    similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    similarity_percentage = round(similarity_score * 100, 2)

    resume_words = set(resume_text.lower().split())
    job_words = set(job_description.lower().split())
    missing_skills = list(job_words - resume_words)

    return similarity_percentage, missing_skills
