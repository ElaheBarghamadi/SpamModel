from django import forms
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm, PasswordResetForm
from .models import Profile


class PersianUserCreationForm(UserCreationForm):
    username = forms.CharField(
        label='نام کاربری',
        max_length=150,
        widget=forms.TextInput(attrs={
            'class': 'form-input',
            'placeholder': 'نام کاربری خود را وارد کنید',
            'autocomplete': 'username',
        }),
        help_text='حداکثر ۱۵۰ کاراکتر. فقط حروف، اعداد و @/./+/-/_ مجاز است.',
    )
    email = forms.EmailField(
        label='ایمیل',
        required=True,
        widget=forms.EmailInput(attrs={
            'class': 'form-input',
            'placeholder': 'ایمیل خود را وارد کنید',
            'autocomplete': 'email',
        }),
    )
    first_name = forms.CharField(
        label='نام',
        max_length=30,
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'form-input',
            'placeholder': 'نام خود را وارد کنید',
        }),
    )
    last_name = forms.CharField(
        label='نام خانوادگی',
        max_length=30,
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'form-input',
            'placeholder': 'نام خانوادگی خود را وارد کنید',
        }),
    )
    password1 = forms.CharField(
        label='رمز عبور',
        widget=forms.PasswordInput(attrs={
            'class': 'form-input',
            'placeholder': 'رمز عبور خود را وارد کنید',
            'autocomplete': 'new-password',
        }),
        help_text='رمز عبور شما نباید شبیه اطلاعات شخصی شما باشد.',
    )
    password2 = forms.CharField(
        label='تکرار رمز عبور',
        widget=forms.PasswordInput(attrs={
            'class': 'form-input',
            'placeholder': 'رمز عبور را دوباره وارد کنید',
            'autocomplete': 'new-password',
        }),
        help_text='برای تأیید، همان رمز عبور قبلی را وارد کنید.',
    )

    class Meta:
        model = User
        fields = ('username', 'email', 'first_name', 'last_name', 'password1', 'password2')

    def clean_email(self):
        email = self.cleaned_data.get('email')
        if User.objects.filter(email=email).exists():
            raise forms.ValidationError('این ایمیل قبلاً ثبت شده است.')
        return email


class PersianLoginForm(AuthenticationForm):
    username = forms.CharField(
        label='نام کاربری',
        widget=forms.TextInput(attrs={
            'class': 'form-input',
            'placeholder': 'نام کاربری',
            'autocomplete': 'username',
            'autofocus': True,
        }),
    )
    password = forms.CharField(
        label='رمز عبور',
        widget=forms.PasswordInput(attrs={
            'class': 'form-input',
            'placeholder': 'رمز عبور',
            'autocomplete': 'current-password',
        }),
    )


class PersianPasswordResetForm(PasswordResetForm):
    email = forms.EmailField(
        label='ایمیل',
        max_length=254,
        widget=forms.EmailInput(attrs={
            'class': 'form-input',
            'placeholder': 'ایمیل ثبت‌نامی خود را وارد کنید',
            'autocomplete': 'email',
        }),
    )


class ProfileUpdateForm(forms.ModelForm):
    first_name = forms.CharField(
        label='نام',
        max_length=30,
        required=False,
        widget=forms.TextInput(attrs={'class': 'form-input'}),
    )
    last_name = forms.CharField(
        label='نام خانوادگی',
        max_length=30,
        required=False,
        widget=forms.TextInput(attrs={'class': 'form-input'}),
    )
    email = forms.EmailField(
        label='ایمیل',
        widget=forms.EmailInput(attrs={'class': 'form-input'}),
    )

    class Meta:
        model = Profile
        fields = ('avatar', 'bio')
        widgets = {
            'bio': forms.Textarea(attrs={'class': 'form-input', 'rows': 3}),
            'avatar': forms.FileInput(attrs={'class': 'form-input'}),
        }
        labels = {
            'avatar': 'تصویر پروفایل',
            'bio': 'بیوگرافی',
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.instance and self.instance.user:
            self.fields['first_name'].initial = self.instance.user.first_name
            self.fields['last_name'].initial = self.instance.user.last_name
            self.fields['email'].initial = self.instance.user.email

    def save(self, commit=True):
        profile = super().save(commit=False)
        profile.user.first_name = self.cleaned_data['first_name']
        profile.user.last_name = self.cleaned_data['last_name']
        profile.user.email = self.cleaned_data['email']
        if commit:
            profile.user.save()
            profile.save()
        return profile
