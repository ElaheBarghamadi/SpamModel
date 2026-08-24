import json
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
from django.contrib.auth.models import User
from django.utils import timezone
from .models import ChatRoom, Message
from .spam_detector import check_message


class ChatConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.room_id = self.scope['url_route']['kwargs']['room_id']
        self.room_group_name = f'chat_{self.room_id}'
        self.user = self.scope['user']

        if self.user.is_anonymous:
            await self.close()
            return

        # بررسی دسترسی کاربر به اتاق
        if not await self.user_has_access():
            await self.close()
            return

        # پیوستن به گروه چت
        await self.channel_layer.group_add(
            self.room_group_name,
            self.channel_name
        )

        await self.accept()

        # ارسال وضعیت آنلاین
        await self.set_user_online(True)
        await self.channel_layer.group_send(
            self.room_group_name,
            {
                'type': 'user_status',
                'username': self.user.username,
                'status': 'online',
            }
        )

    async def disconnect(self, close_code):
        if hasattr(self, 'room_group_name'):
            # ارسال وضعیت آفلاین
            await self.set_user_online(False)
            await self.channel_layer.group_send(
                self.room_group_name,
                {
                    'type': 'user_status',
                    'username': self.user.username,
                    'status': 'offline',
                }
            )

            # ترک گروه چت
            await self.channel_layer.group_discard(
                self.room_group_name,
                self.channel_name
            )

    async def receive(self, text_data):
        try:
            data = json.loads(text_data)
            message_type = data.get('type', 'message')

            if message_type == 'message':
                content = data.get('content', '').strip()
                force_send = data.get('force_send', False)
                if not content:
                    return

                # ✅ بررسی اسپم و محتوای نامناسب
                spam_result = await self.analyze_spam(content)

                if spam_result['is_harmful'] and not force_send:
                    # ارسال هشدار به فرستنده
                    await self.send(text_data=json.dumps({
                        'type': 'spam_warning',
                        'warning': {
                            'message': spam_result['warning_message'],
                            'warning_type': spam_result['warning_type'],
                            'label': spam_result['label'],
                            'confidence': round(spam_result['confidence'] * 100),
                            'original_content': content,
                        }
                    }))
                    return

                # ذخیره پیام در دیتابیس
                message = await self.save_message(content)

                # ارسال پیام به گروه
                await self.channel_layer.group_send(
                    self.room_group_name,
                    {
                        'type': 'chat_message',
                        'message': {
                            'id': message.id,
                            'content': content,
                            'sender': self.user.username,
                            'sender_name': await self.get_display_name(),
                            'timestamp': message.timestamp.strftime('%H:%M'),
                            'date': message.timestamp.strftime('%Y/%m/%d'),
                            'is_sender': False,
                        }
                    }
                )

            elif message_type == 'typing':
                await self.channel_layer.group_send(
                    self.room_group_name,
                    {
                        'type': 'typing_indicator',
                        'username': self.user.username,
                        'is_typing': data.get('is_typing', False),
                    }
                )

            elif message_type == 'read':
                await self.mark_messages_read()
                await self.channel_layer.group_send(
                    self.room_group_name,
                    {
                        'type': 'messages_read',
                        'username': self.user.username,
                    }
                )

        except json.JSONDecodeError:
            pass

    async def chat_message(self, event):
        message = event['message']
        message['is_sender'] = (message['sender'] == self.user.username)

        await self.send(text_data=json.dumps({
            'type': 'message',
            'message': message,
        }))

    async def typing_indicator(self, event):
        if event['username'] != self.user.username:
            await self.send(text_data=json.dumps({
                'type': 'typing',
                'username': event['username'],
                'is_typing': event['is_typing'],
            }))

    async def user_status(self, event):
        await self.send(text_data=json.dumps({
            'type': 'status',
            'username': event['username'],
            'status': event['status'],
        }))

    async def messages_read(self, event):
        if event['username'] != self.user.username:
            await self.send(text_data=json.dumps({
                'type': 'read',
                'username': event['username'],
            }))

    @database_sync_to_async
    def user_has_access(self):
        try:
            room = ChatRoom.objects.get(id=self.room_id)
            return room.participants.filter(id=self.user.id).exists()
        except ChatRoom.DoesNotExist:
            return False

    @database_sync_to_async
    def save_message(self, content):
        room = ChatRoom.objects.get(id=self.room_id)
        message = Message.objects.create(
            room=room,
            sender=self.user,
            content=content,
        )
        room.save()  # بروزرسانی updated_at
        return message

    @database_sync_to_async
    def get_display_name(self):
        profile = self.user.profile
        return profile.display_name

    @database_sync_to_async
    def set_user_online(self, status):
        self.user.profile.online = status
        self.user.profile.save()

    @database_sync_to_async
    def mark_messages_read(self):
        room = ChatRoom.objects.get(id=self.room_id)
        room.messages.filter(is_read=False).exclude(sender=self.user).update(is_read=True)

    @database_sync_to_async
    def analyze_spam(self, content):
        """تحلیل محتوای پیام برای تشخیص اسپم"""
        return check_message(content)


class NotificationConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.user = self.scope['user']

        if self.user.is_anonymous:
            await self.close()
            return

        self.notification_group_name = f'notifications_{self.user.id}'

        await self.channel_layer.group_add(
            self.notification_group_name,
            self.channel_name
        )

        await self.accept()

    async def disconnect(self, close_code):
        if hasattr(self, 'notification_group_name'):
            await self.channel_layer.group_discard(
                self.notification_group_name,
                self.channel_name
            )

    async def send_notification(self, event):
        await self.send(text_data=json.dumps({
            'type': 'notification',
            'data': event['data'],
        }))
