# Generated by Django 3.2.7 on 2021-12-07 02:58

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('users', '0008_alter_user_supervisor'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='user',
            name='schoolID',
        ),
    ]