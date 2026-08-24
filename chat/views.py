from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.contrib import messages
from django.http import JsonResponse
from django.db.models import Q, Max, Subquery, OuterRef
from django.db.models.functions import Greatest
from django.views.decorators.csrf import csrf_exempt
from .models import ChatRoom, Message
from .spam_detector import check_message


@login_required
def home_view(request):
    """صفحه اصلی - لیست چت‌ها"""
    user = request.user
    
    # دریافت اتاق‌های چت کاربر
    chat_rooms = ChatRoom.objects.filter(
        participants=user
    ).annotate(
        last_message_time=Max('messages__timestamp')
    ).order_by('-last_message_time')

    # اضافه کردن اطلاعات اضافی به هر اتاق
    rooms_data = []
    for room in chat_rooms:
        other_user = room.get_other_participant(user)
        last_msg = room.last_message
        unread_count = room.messages.filter(is_read=False).exclude(sender=user).count()
        
        rooms_data.append({
            'room': room,
            'other_user': other_user,
            'last_message': last_msg,
            'unread_count': unread_count,
        })

    context = {
        'rooms_data': rooms_data,
        'user': user,
    }
    return render(request, 'chat/home.html', context)


@login_required
def room_view(request, room_id):
    """صفحه چت"""
    room = get_object_or_404(ChatRoom, id=room_id)
    
    # بررسی دسترسی
    if not room.participants.filter(id=request.user.id).exists():
        messages.error(request, 'شما به این اتاق چت دسترسی ندارید.')
        return redirect('chat:home')

    other_user = room.get_other_participant(request.user)
    
    # علامت‌گذاری پیام‌ها به عنوان خوانده شده
    room.messages.filter(is_read=False).exclude(sender=request.user).update(is_read=True)
    
    # دریافت پیام‌ها
    chat_messages = room.messages.select_related('sender').all()[:50]

    context = {
        'room': room,
        'other_user': other_user,
        'messages': chat_messages,
    }
    return render(request, 'chat/room.html', context)


@login_required
def start_chat(request, username):
    """شروع چت با کاربر"""
    other_user = get_object_or_404(User, username=username)
    
    if other_user == request.user:
        messages.error(request, 'نمی‌توانید با خودتان چت کنید!')
        return redirect('chat:home')

    # بررسی وجود اتاق چت قبلی
    existing_room = ChatRoom.objects.filter(
        room_type='private',
        participants=request.user
    ).filter(
        participants=other_user
    ).first()

    if existing_room:
        return redirect('chat:room', room_id=existing_room.id)

    # ایجاد اتاق چت جدید
    room = ChatRoom.objects.create(
        name=f'{request.user.username}-{other_user.username}',
        room_type='private',
    )
    room.participants.add(request.user, other_user)

    return redirect('chat:room', room_id=room.id)


@login_required
def user_list_view(request):
    """لیست کاربران برای شروع چت"""
    users = User.objects.exclude(id=request.user.id).select_related('profile')
    
    search_query = request.GET.get('q', '')
    if search_query:
        users = users.filter(
            Q(username__icontains=search_query) |
            Q(first_name__icontains=search_query) |
            Q(last_name__icontains=search_query)
        )

    context = {
        'users': users,
        'search_query': search_query,
    }
    return render(request, 'chat/user_list.html', context)


@login_required
def search_messages(request):
    """جستجو در پیام‌ها"""
    query = request.GET.get('q', '')
    room_id = request.GET.get('room', None)
    
    if not query:
        return JsonResponse({'messages': []})

    messages_qs = Message.objects.filter(
        room__participants=request.user,
        content__icontains=query
    ).select_related('sender', 'room')

    if room_id:
        messages_qs = messages_qs.filter(room_id=room_id)

    messages_qs = messages_qs.order_by('-timestamp')[:20]

    results = [{
        'id': msg.id,
        'content': msg.content,
        'sender': msg.sender.username,
        'room_id': msg.room.id,
        'timestamp': msg.timestamp.strftime('%H:%M'),
        'date': msg.timestamp.strftime('%Y/%m/%d'),
    } for msg in messages_qs]

    return JsonResponse({'messages': results})


@login_required
def test_spam_view(request):
    """صفحه تست تشخیص اسپم"""
    result = None
    test_text = ''

    if request.method == 'POST':
        test_text = request.POST.get('text', '')
        if test_text:
            result = check_message(test_text)

    context = {
        'result': result,
        'test_text': test_text,
    }
    return render(request, 'chat/test_spam.html', context)


@csrf_exempt
def check_spam_api(request):
    """API endpoint برای بررسی اسپم"""
    if request.method == 'POST':
        import json
        try:
            data = json.loads(request.body)
            text = data.get('text', '')
            if text:
                result = check_message(text)
                return JsonResponse(result)
            return JsonResponse({'error': 'متن ارسال نشده'}, status=400)
        except json.JSONDecodeError:
            return JsonResponse({'error': 'درخواست نامعتبر'}, status=400)
    return JsonResponse({'error': 'فقط POST مجاز است'}, status=405)
