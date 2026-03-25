from layer4_injection.base_injector import BaseInjector
from layer4_injection.desktop_injector import DesktopInjector
from layer4_injection.web_injector import WebInjector
from core.exceptions import InjectionError


class InjectorFactory:
    @staticmethod
    def create(target: str, config: dict) -> BaseInjector:
        if target == "desktop":
            return DesktopInjector(config)
        elif target == "web":
            return WebInjector(config)
        elif target == "json_output":
            from layer4_injection.json_output_injector import JsonOutputInjector
            return JsonOutputInjector(config)
        else:
            raise InjectionError(f"Unknown injection target: {target}. Use 'desktop', 'web', or 'json_output'.")
