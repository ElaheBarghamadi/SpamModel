from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from detector import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.home, name='home'),
    path('train/', views.train_model, name='train'),
    path('test/', views.test_text, name='test_text'),
    path('test-file/', views.test_file, name='test_file'),
    path('api/predict/', views.api_predict, name='api_predict'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
