import sys
from pathlib import Path
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import json

# Add parent directory to Python path (better to handle this differently in production)
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from main import BookRecommender

# Initialize the recommender once when Django starts
recommender = BookRecommender()
recommender.load_models()  # Assuming this loads your model data

@csrf_exempt
@require_http_methods(["POST"])
def get_recs(request):
    try:
        # Parse JSON data from request body
        data = json.loads(request.body)
        titles = data.get('titles', [])
        
        # Validate input
        if not titles:
            return JsonResponse({
                "status": "error",
                "message": "No book titles provided"
            }, status=400)
        
        # Get recommendations (assuming you have a public method for this)
        # You may need to adjust this method name based on your actual BookRecommender class
        recs = recommender._similar_books__name(titles)  # Changed to a more appropriate method name
        
        return JsonResponse({
            "status": "success",
            "recommendations": recs,
            "input_books": titles
        })
    except json.JSONDecodeError:
        return JsonResponse({
            "status": "error",
            "message": "Invalid JSON data"
        }, status=400)
    except Exception as e:
        return JsonResponse({
            "status": "error",
            "message": str(e)
        }, status=500)