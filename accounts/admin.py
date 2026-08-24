from django.contrib import admin
from .models import Profile


@admin.register(Profile)
class ProfileAdmin(admin.ModelAdmin):
    list_display = ('user', 'online', 'last_seen')
    list_filter = ('online',)
    search_fields = ('user__username', 'user__email')
