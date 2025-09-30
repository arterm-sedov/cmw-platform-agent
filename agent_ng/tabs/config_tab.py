"""
Config Tab for Comindware connection settings.

Fields: Platform URL, Username, Password (masked).
Wiring: Save to BrowserState (per-browser), Load from BrowserState.
Supports i18n.
"""

from collections.abc import Callable
from contextlib import suppress
import logging
import os
from typing import Any

import gradio as gr

# Import for translations
from agent_ng.i18n_translations import get_translation_key
from agent_ng.session_manager import set_session_config


class ConfigTab:
    """Configuration tab component for Comindware Platform settings"""

    def __init__(
        self,
        event_handlers: dict[str, Callable],
        language: str = "en",
        i18n_instance: gr.I18n | None = None,
    ) -> None:
        self.event_handlers = event_handlers
        self.components: dict[str, Any] = {}
        self.main_app = None  # Reference to main app if needed later
        self.language = language
        self.i18n = i18n_instance
        # No server-side cache here; we store per-session config via SessionManager

    def create_tab(self) -> tuple[gr.TabItem, dict[str, Any]]:
        """
        Build config tab UI (URL, username, password).

        Returns (TabItem, components_dict).
        """
        logging.getLogger(__name__).info(
            "✅ ConfigTab: Creating configuration interface..."
        )

        # Config tab should be shown when CMW_USE_DOTENV=false (default)
        # Hide when CMW_USE_DOTENV=true
        use_dotenv_flag = os.environ.get("CMW_USE_DOTENV", "true").lower() in (
            "1",
            "true",
            "yes",
        )
        if use_dotenv_flag:
            return None, self.components

        with gr.TabItem(self._get_translation("tab_config"), id="config") as tab:
            # Create configuration interface
            self._create_config_interface()

            # Wire events
            self._connect_events()

            # Auto-load BrowserState into fields on render (and when it changes)
            gr.on(
                triggers=[self.components["config_state"].change],
                inputs=[self.components["config_state"]],
                outputs=[
                    self.components["platform_url"],
                    self.components["username"],
                    self.components["password"],
                ],
            )(self._load_from_state)

        logging.getLogger(__name__).info(
            "✅ ConfigTab: Successfully created with components and wiring"
        )
        return tab, self.components

    def _create_config_interface(self) -> None:
        """Create the configuration form with consistent styling"""

        url_init, login_init, password_init = "", "", ""

        # Config state in browser localStorage (shared across tabs in same browser)
        self.components["config_state"] = gr.BrowserState(
            {
                "url": url_init,
                "username": login_init,
                "password": password_init,
            },
            storage_key="cmw_config_v1",
        )

        # Use the same card-like styling used elsewhere (model-card)
        with gr.Column(scale=1, min_width=400, elem_classes=["model-card"]):
            gr.Markdown(
                f"### {self._get_translation('config_title')}",
                elem_classes=["llm-selection-title"],
            )

            # Platform URL
            self.components["platform_url"] = gr.Textbox(
                label=self._get_translation("config_platform_url"),
                placeholder="https://your-comindware-host",
                value=url_init,
                lines=1,
                max_lines=1,
                show_copy_button=False,
            )

            # Username
            self.components["username"] = gr.Textbox(
                label=self._get_translation("config_username"),
                value=login_init,
                lines=1,
                max_lines=1,
                show_copy_button=False,
            )

            self.components["password"] = gr.Textbox(
                label=self._get_translation("config_password"),
                type="password",  # See Gradio Textbox type parameter
                value=password_init,
                lines=1,
                max_lines=1,
                show_copy_button=False,
            )

            with gr.Row(equal_height=True):
                self.components["save_btn"] = gr.Button(
                    self._get_translation("config_save_button"),
                    variant="primary",
                    elem_classes=["cmw-button"],
                )
                self.components["load_btn"] = gr.Button(
                    self._get_translation("config_load_button"),
                    variant="secondary",
                    elem_classes=["cmw-button"],
                )
                self.components["clear_storage_btn"] = gr.Button(
                    self._get_translation("config_clear_storage_button"),
                    variant="secondary",
                    elem_classes=["cmw-button"],
                )

            # Status area (not used as output to avoid version mismatches)
            self.components["config_status_display"] = gr.Markdown("")

    def _connect_events(self) -> None:
        """Wire events for Save/Load configuration."""
        logging.getLogger(__name__).debug("🔗 ConfigTab: Connecting events...")

        # Save to browser state (and update runtime env)
        self.components["save_btn"].click(
            fn=self._save_to_state,
            inputs=[
                self.components["platform_url"],
                self.components["username"],
                self.components["password"],
                self.components["config_state"],
            ],
            outputs=[self.components["config_state"]],
        )

        # Load from browser state
        self.components["load_btn"].click(
            fn=self._load_from_state,
            inputs=[self.components["config_state"]],
            outputs=[
                self.components["platform_url"],
                self.components["username"],
                self.components["password"],
            ],
        )

        # Clear browser storage and reset fields
        self.components["clear_storage_btn"].click(
            fn=self._clear_browser_storage,
            inputs=[self.components["config_state"]],
            outputs=[
                self.components["config_state"],
                self.components["platform_url"],
                self.components["username"],
                self.components["password"],
            ],
        )

    # Event handlers
    def _save_to_state(
        self,
        url: str,
        username: str,
        password: str,
        current_state: dict | None,
        request: gr.Request | None = None,
    ) -> dict:
        """Save configuration into browser state and update process env."""
        try:
            url = (url or "").strip()
            username = (username or "").strip()
            password = (password or "").strip()

            # Prepare new state dict
            new_state = {
                "url": url,
                "username": username,
                "password": password,
            }

            # Determine accurate session id
            session_id = None
            try:
                if (
                    request
                    and hasattr(request, "session_hash")
                    and request.session_hash
                ):
                    session_id = f"gradio_{request.session_hash}"
            except Exception:
                session_id = None

            if not session_id and (
                self.main_app
                and hasattr(self.main_app, "session_manager")
                and hasattr(self.main_app.session_manager, "get_last_active_session_id")
            ):
                session_id = (
                    self.main_app.session_manager.get_last_active_session_id()  # type: ignore[attr-defined]
                )

            if session_id:
                set_session_config(session_id, new_state)
                try:
                    masked = {
                        "url_present": bool(url),
                        "username_len": len(username),
                        "password_len": len(password),
                    }
                    logging.getLogger(__name__).debug(
                        "🔐 ConfigTab.save -> session=%s stored=%s",
                        session_id,
                        masked,
                    )
                except Exception:
                    logging.getLogger(__name__).debug(
                        "🔐 ConfigTab.save -> debug logging failed",
                        exc_info=True,
                    )

            with suppress(Exception):
                gr.Info(self._get_translation("config_save_success_session"))
        except Exception as e:
            logging.getLogger(__name__).exception("Save to browser state failed")
            with suppress(Exception):
                gr.Warning(self._get_translation("config_save_error") + f"\n\n{str(e)}")
            return current_state or {}
        else:
            return new_state

    def _load_from_state(
        self, state: Any, request: gr.Request | None = None
    ) -> tuple[Any, Any, Any]:
        """Load values from browser state only and update fields."""
        try:
            # Normalize state across gradio versions (may come as tuple or dict)
            if isinstance(state, tuple) and len(state) > 0:
                state = state[0]
            if not isinstance(state, dict):
                state = {}

            # If state has non-empty values, use them
            if any(state.get(k) for k in ("url", "username", "password")):
                url = state.get("url", "") or ""
                login = state.get("username", "") or ""
                pwd = state.get("password", "") or ""
                # Also propagate BrowserState snapshot into per-session store
                # for backend
                try:
                    session_id = None
                    try:
                        if (
                            request
                            and hasattr(request, "session_hash")
                            and request.session_hash
                        ):
                            session_id = f"gradio_{request.session_hash}"
                    except Exception:
                        session_id = None

                    if not session_id and (
                        self.main_app
                        and hasattr(self.main_app, "session_manager")
                        and hasattr(
                            self.main_app.session_manager,
                            "get_last_active_session_id",
                        )
                    ):
                        session_id = (
                            self.main_app.session_manager.get_last_active_session_id()  # type: ignore[attr-defined]
                        )

                    if session_id:
                        set_session_config(
                            session_id,
                            {"url": url, "username": login, "password": pwd},
                        )
                        logging.getLogger(__name__).debug(
                            (
                                "🔄 ConfigTab.load -> propagated "
                                "BrowserState to session=%s"
                            ),
                            session_id,
                        )
                except Exception:
                    logging.getLogger(__name__).debug(
                        (
                            "⚠️ ConfigTab.load -> failed to propagate "
                            "BrowserState to session store"
                        ),
                        exc_info=True,
                    )
            else:
                url, login, pwd = "", "", ""

            with suppress(Exception):
                gr.Info(self._get_translation("config_load_success"))
            return (
                gr.update(value=url),
                gr.update(value=login),
                gr.update(value=pwd),
            )
        except Exception as e:
            logging.getLogger(__name__).exception("Load from browser state failed")
            with suppress(Exception):
                gr.Warning(self._get_translation("config_load_error") + f"\n\n{str(e)}")
            return (
                gr.update(),
                gr.update(),
                gr.update(),
            )

    # Removed .env loading: rely solely on browser state

    def _clear_browser_storage(self, state: Any) -> tuple[dict, Any, Any, Any]:
        """Clear browser-persisted state and reset input fields."""
        try:
            new_state: dict = {}

            with suppress(Exception):
                gr.Info(self._get_translation("config_clear_success"))
            return (
                new_state,
                gr.update(value=""),
                gr.update(value=""),
                gr.update(value=""),
            )
        except Exception as e:
            logging.getLogger(__name__).exception("Clear browser storage failed")
            with suppress(Exception):
                gr.Warning(self._get_translation("config_clear_error") + f"\n\n{str(e)}")
            # Return original state if clear failed
            return state or {}, gr.update(), gr.update(), gr.update()

    def set_main_app(self, main_app: Any) -> None:
        """Set reference to main app for future integration needs."""
        self.main_app = main_app

    def get_components(self) -> dict[str, Any]:
        """Return created components."""
        return self.components

    # No get_current_config needed; backend reads from SessionManager per session

    def _get_translation(self, key: str) -> str:
        """Get a translation for a specific key (consistent with other tabs)"""
        # Always use direct translation for now to avoid i18n metadata issues
        return get_translation_key(key, self.language)
