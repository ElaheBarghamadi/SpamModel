from django.db import models
from django.contrib.auth.models import User


class ChatRoom(models.Model):
    ROOM_TYPES = (
        ('private', 'خصوصی'),
        ('group', 'گروهی'),
    )
    
    name = models.CharField(max_length=255, verbose_name='نام اتاق')
    room_type = models.CharField(max_length=10, choices=ROOM_TYPES, default='private', verbose_name='نوع اتاق')
    participants = models.ManyToManyField(User, related_name='chat_rooms', verbose_name='شرکت‌کنندگان')
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='زمان ایجاد')
    updated_at = models.DateTimeField(auto_now=True, verbose_name='آخرین بروزرسانی')

    class Meta:
        verbose_name = 'اتاق چت'
        verbose_name_plural = 'اتاق‌های چت'
        ordering = ['-updated_at']

    def __str__(self):
        return self.name

    @property
    def last_message(self):
        return self.messages.order_by('-timestamp').first()

    def get_other_participant(self, user):
        """برای چت خصوصی، کاربر مقابل را برمی‌گرداند"""
        return self.participants.exclude(id=user.id).first()


class Message(models.Model):
    MESSAGE_TYPES = (
        ('text', 'متن'),
        ('image', 'تصویر'),
        ('file', 'فایل'),
    )
    
    room = models.ForeignKey(ChatRoom, on_delete=models.CASCADE, related_name='messages', verbose_name='اتاق')
    sender = models.ForeignKey(User, on_delete=models.CASCADE, related_name='sent_messages', verbose_name='فرستنده')
    content = models.TextField(verbose_name='محتوا')
    message_type = models.CharField(max_length=10, choices=MESSAGE_TYPES, default='text', verbose_name='نوع پیام')
    timestamp = models.DateTimeField(auto_now_add=True, verbose_name='زمان ارسال')
    is_read = models.BooleanField(default=False, verbose_name='خوانده شده')

    class Meta:
        verbose_name = 'پیام'
        verbose_name_plural = 'پیام‌ها'
        ordering = ['timestamp']

    def __str__(self):
        return f'{self.sender.username}: {self.content[:50]}'
