from django.db import models
from django.contrib.auth.models import User
from django.db.models.signals import post_save
from django.dispatch import receiver


class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile', verbose_name='کاربر')
    avatar = models.ImageField(upload_to='avatars/', null=True, blank=True, verbose_name='آواتار')
    bio = models.TextField(max_length=500, blank=True, default='', verbose_name='بیوگرافی')
    online = models.BooleanField(default=False, verbose_name='آنلاین')
    last_seen = models.DateTimeField(auto_now=True, verbose_name='آخرین بازدید')

    class Meta:
        verbose_name = 'پروفایل'
        verbose_name_plural = 'پروفایل‌ها'

    def __str__(self):
        return f'پروفایل {self.user.username}'

    @property
    def display_name(self):
        if self.user.first_name:
            return f'{self.user.first_name} {self.user.last_name}'.strip()
        return self.user.username


@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    if created:
        Profile.objects.create(user=instance)


@receiver(post_save, sender=User)
def save_user_profile(sender, instance, **kwargs):
    instance.profile.save()
