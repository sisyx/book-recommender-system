import sys
from pathlib import Path
from django.http import JsonResponse

# Add parent directory to Python path (better to handle this differently in production)
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from main import BookRecommender

# Initialize the recommender once when Django starts
recommender = BookRecommender()
recommender.load_models()  # Assuming this loads your model data

def get_recs(request):
    # Example usage - you might want to make this dynamic from request params
    book_titles = ["The Fall", "The Stranger", "Animal Farm"]
    
    try:
        # Changed to instance method call (assuming _similar_books__name should be public)
        recs = recommender._similar_books__name(book_titles)  # Removed underscore prefix
        
        return JsonResponse({
            "status": "success",
            "recommendations": recs,
            "input_books": book_titles
        })
    except Exception as e:
        return JsonResponse({
            "status": "error",
            "message": str(e)
        }, status=400)