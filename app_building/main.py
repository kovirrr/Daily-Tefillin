from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivymd.app import MDApp
from kivymd.uix.button import MDRaisedButton
from kivymd.uix.label import MDLabel
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.toolbar import MDTopAppBar
from kivy.uix.camera import Camera

KV = '''
ScreenManager:
    HomeScreen:
    CameraScreen:
    StreakScreen:

<HomeScreen>:
    name: "home"
    MDBoxLayout:
        orientation: 'vertical'
        spacing: dp(20)
        padding: dp(20)

        MDTopAppBar:
            title: "Tefillin Checker"
            elevation: 4

        MDLabel:
            text: "Welcome!"
            font_style: "H4"
            halign: "center"

        MDRaisedButton:
            text: "Check Tefillin"
            pos_hint: {"center_x": 0.5}
            size_hint_x: 0.8
            on_press: app.root.current = "camera"

        MDRaisedButton:
            text: "View Streak"
            pos_hint: {"center_x": 0.5}
            size_hint_x: 0.8
            on_press: app.root.current = "streak"

<CameraScreen>:
    name: "camera"
    MDBoxLayout:
        orientation: 'vertical'

        MDTopAppBar:
            title: "Camera"
            elevation: 4
            left_action_items: [["arrow-left", lambda x: app.root.current = "home"]]

        Camera:
            id: cam
            resolution: (640, 480)
            play: True

        MDRaisedButton:
            text: "Back to Home"
            pos_hint: {"center_x": 0.5}
            size_hint_x: 0.8
            on_press: app.root.current = "home"

<StreakScreen>:
    name: "streak"
    MDBoxLayout:
        orientation: 'vertical'
        spacing: dp(20)
        padding: dp(20)

        MDTopAppBar:
            title: "Daily Streak"
            elevation: 4
            left_action_items: [["arrow-left", lambda x: app.root.current = "home"]]

        MDLabel:
            id: streak_label
            text: "Streak: 0"
            font_style: "H4"
            halign: "center"

        MDRaisedButton:
            text: "Back to Home"
            pos_hint: {"center_x": 0.5}
            size_hint_x: 0.8
            on_press: app.root.current = "home"
'''

class HomeScreen(Screen):
    pass

class CameraScreen(Screen):
    pass

class StreakScreen(Screen):
    pass

class TefillinApp(MDApp):
    def build(self):
        self.theme_cls.primary_palette = "Blue"
        return Builder.load_string(KV)

if __name__ == '__main__':
    TefillinApp().run()
