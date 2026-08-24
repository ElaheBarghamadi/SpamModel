from django.contrib import admin
from .models import ChatRoom, Message


@admin.register(ChatRoom)
class ChatRoomAdmin(admin.ModelAdmin):
    list_display = ('name', 'room_type', 'created_at', 'updated_at')
    list_filter = ('room_type',)
    search_fields = ('name',)
    filter_horizontal = ('participants',)


@admin.register(Message)
class MessageAdmin(admin.ModelAdmin):
    list_display = ('sender', 'room', 'content', 'timestamp', 'is_read')
    list_filter = ('is_read', 'message_type')
    search_fields = ('content', 'sender__username')
    date_hierarchy = 'timestamp'
