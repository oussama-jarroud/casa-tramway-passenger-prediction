# casa_tramway_project/urls.py
from django.contrib import admin
from django.urls import path, include # <-- Ajoute 'include'

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('prediction_app.urls')), # <-- Ajoute ceci pour inclure les URLs de ton app
]

# Pour servir les fichiers médias en développement
from django.conf import settings
from django.conf.urls.static import static
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)