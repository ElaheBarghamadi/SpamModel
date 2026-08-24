from django.urls import path
from . import views

app_name = 'accounts'

urlpatterns = [
    path('register/', views.register_view, name='register'),
    path('login/', views.PersianLoginView.as_view(), name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('profile/', views.profile_view, name='profile'),
    path('password-reset/', views.PersianPasswordResetView.as_view(), name='password_reset'),
    path('password-reset/done/', views.PersianPasswordResetDoneView.as_view(), name='password_reset_done'),
    path('password-reset/confirm/<uidb64>/<token>/', views.PersianPasswordResetConfirmView.as_view(), name='password_reset_confirm'),
    path('password-reset/complete/', views.PersianPasswordResetCompleteView.as_view(), name='password_reset_complete'),
]
