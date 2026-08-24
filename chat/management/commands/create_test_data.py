from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from chat.models import ChatRoom, Message


class Command(BaseCommand):
    help = 'ایجاد داده‌های آزمایشی'

    def handle(self, *args, **options):
        # ایجاد کاربران آزمایشی
        users_data = [
            {'username': 'ali', 'email': 'ali@example.com', 'first_name': 'علی', 'last_name': 'محمدی', 'password': 'test1234'},
            {'username': 'sara', 'email': 'sara@example.com', 'first_name': 'سارا', 'last_name': 'احمدی', 'password': 'test1234'},
            {'username': 'reza', 'email': 'reza@example.com', 'first_name': 'رضا', 'last_name': 'کریمی', 'password': 'test1234'},
            {'username': 'mina', 'email': 'mina@example.com', 'first_name': 'مینا', 'last_name': 'رضایی', 'password': 'test1234'},
        ]

        users = []
        for data in users_data:
            user, created = User.objects.get_or_create(
                username=data['username'],
                defaults={
                    'email': data['email'],
                    'first_name': data['first_name'],
                    'last_name': data['last_name'],
                }
            )
            if created:
                user.set_password(data['password'])
                user.save()
                self.stdout.write(self.style.SUCCESS(f'کاربر {user.username} ایجاد شد'))
            users.append(user)

        # ایجاد اتاق‌های چت آزمایشی
        room1, _ = ChatRoom.objects.get_or_create(
            name='ali-sara',
            defaults={'room_type': 'private'}
        )
        room1.participants.add(users[0], users[1])

        room2, _ = ChatRoom.objects.get_or_create(
            name='ali-reza',
            defaults={'room_type': 'private'}
        )
        room2.participants.add(users[0], users[2])

        # ایجاد پیام‌های آزمایشی
        messages_data = [
            {'room': room1, 'sender': users[0], 'content': 'سلام سارا! خوبی؟'},
            {'room': room1, 'sender': users[1], 'content': 'سلام علی! ممنون، تو خوبی؟'},
            {'room': room1, 'sender': users[0], 'content': 'آره خوبم. چه خبر؟'},
            {'room': room1, 'sender': users[1], 'content': 'هیچی خاصی، داشتم درس می‌خوندم'},
            {'room': room2, 'sender': users[0], 'content': 'رضا سلام!'},
            {'room': room2, 'sender': users[2], 'content': 'سلام داداش! چطوری؟'},
        ]

        for data in messages_data:
            Message.objects.get_or_create(
                room=data['room'],
                sender=data['sender'],
                content=data['content'],
            )

        self.stdout.write(self.style.SUCCESS('داده‌های آزمایشی با موفقیت ایجاد شدند'))
        self.stdout.write(self.style.SUCCESS('نام کاربری و رمز عبور برای همه کاربران: test1234'))