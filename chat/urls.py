from django.urls import path
from . import views

app_name = 'chat'

urlpatterns = [
    path('', views.home_view, name='home'),
    path('room/<int:room_id>/', views.room_view, name='room'),
    path('start/<str:username>/', views.start_chat, name='start_chat'),
    path('users/', views.user_list_view, name='user_list'),
    path('search/', views.search_messages, name='search_messages'),
    path('test-spam/', views.test_spam_view, name='test_spam'),
    path('api/check-spam/', views.check_spam_api, name='check_spam_api'),
]
