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
import pandas as pd

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
        # Filled in _create_config_interface; tests may assign before calling handlers.
        self._config_llm_providers: list[str] = []
        self._llm_keys_df_columns: list[str] = []
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

        logging.getLogger(__name__).info(
            "✅ ConfigTab: Successfully created with components and wiring"
        )
        return tab, self.components

    def _create_config_interface(self) -> None:
        """Create the configuration form with consistent styling"""

        # Default provider from env/config at startup (for dropdown initial value only)
        try:
            from agent_ng.agent_config import get_llm_settings

            llm_settings = get_llm_settings()
            llm_provider_init = llm_settings.get("default_provider", "openrouter")
        except ImportError:
            llm_provider_init = os.environ.get("AGENT_PROVIDER", "openrouter")

        # BrowserState default MUST be all empty strings. Do NOT bake env defaults
        # here — demo.load() fires before onMount() reads localStorage, so it
        # receives the default value. If provider is pre-filled, has_any_value
        # becomes True and url/username/password get cleared.
        self.components["config_state"] = gr.BrowserState(
            {
                "url": "",
                "username": "",
                "password": "",
                "llm_provider_override": "",
                "llm_api_key_override": "",
                "llm_provider_api_keys": {},
            },
            storage_key="cmw_config_v1",
        )

        # Textbox initial values — always empty, BrowserState handles persistence
        url_init = ""
        login_init = ""
        password_init = ""

        # Use the same card-like styling used elsewhere (model-card)
        with gr.Column(scale=1, min_width=400, elem_classes=["model-card"]):
            gr.Markdown(
                f"### {self._get_translation('config_title')}",
                elem_classes=["llm-selection-title"],
            )
            gr.Markdown(self._get_translation("config_help"))

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

            gr.Markdown("---")
            gr.Markdown(f"**{self._get_translation('config_llm_section')}**")

            # Filter providers by env allowlist just like the model selector.
            # Fallback list mirrors LLMProvider enum; overridden at runtime by
            # llm_manager.get_available_providers() below.
            llm_providers = [
                "gemini",
                "groq",
                "huggingface",
                "openai",
                "openrouter",
                "polza",
                "mistral",
                "gigachat",
            ]
            try:
                if (
                    self.main_app
                    and hasattr(self.main_app, "llm_manager")
                    and self.main_app.llm_manager
                ):
                    available = self.main_app.llm_manager.get_available_providers()
                    if available:
                        llm_providers = sorted(available)
            except Exception:
                logging.getLogger(__name__).debug(
                    "ConfigTab: fallback static provider list "
                    "(llm_manager unavailable)",
                    exc_info=True,
                )

            self._config_llm_providers = list(llm_providers)
            col_provider = self._get_translation("config_llm_provider_label")
            col_key = self._get_translation("config_llm_api_key_label")
            self._llm_keys_df_columns = [col_provider, col_key]

            initial_llm_dropdown = (
                llm_provider_init
                if llm_provider_init in self._config_llm_providers
                else ""
            )

            init_df = pd.DataFrame(
                [[p, ""] for p in self._config_llm_providers],
                columns=self._llm_keys_df_columns,
            )
            row_count_llm = len(self._config_llm_providers)

            self.components["llm_provider_override"] = gr.Dropdown(
                label=self._get_translation("config_llm_active_provider_label"),
                choices=["", *self._config_llm_providers],
                value=initial_llm_dropdown,
            )

            self.components["llm_provider_keys_table"] = gr.Dataframe(
                value=init_df,
                label=self._get_translation("config_llm_api_keys_table_label"),
                interactive=True,
                row_count=(row_count_llm, "fixed"),
                col_count=(2, "fixed"),
                static_columns=[0],
                datatype=["str", "str"],
                max_height=min(520, 80 + row_count_llm * 36),
                wrap=True,
            )
            gr.Markdown(self._get_translation("config_llm_api_keys_table_help"))

            gr.Markdown(
                "*" + self._get_translation("config_llm_empty_means_default") + "*"
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
                self.components["llm_provider_override"],
                self.components["llm_provider_keys_table"],
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
                self.components["llm_provider_override"],
                self.components["llm_provider_keys_table"],
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
                self.components["llm_provider_override"],
                self.components["llm_provider_keys_table"],
            ],
        )

    @staticmethod
    def _normalize_llm_provider_api_keys(browser_state: dict | None) -> dict[str, str]:
        """Build canonical provider-to-API-key mapping from BrowserState."""
        if not isinstance(browser_state, dict):
            return {}
        raw = browser_state.get("llm_provider_api_keys")
        mp: dict[str, str] = {}
        if isinstance(raw, dict):
            for pk, vk in raw.items():
                if not isinstance(pk, str):
                    continue
                k_strip = pk.strip()
                if not k_strip:
                    continue
                val = vk if isinstance(vk, str) else ""
                mp[k_strip] = val.strip()
        # Single-key legacy rows (saved before multi-key map existed)
        leg_p = (browser_state.get("llm_provider_override") or "").strip()
        leg_k = (browser_state.get("llm_api_key_override") or "").strip()
        if leg_p and leg_k and leg_p not in mp:
            mp[leg_p] = leg_k
        return mp

    @staticmethod
    def _parse_llm_keys_dataframe(
        data: Any,
        providers: list[str],
    ) -> dict[str, str]:
        """Map each configured provider row to edited API key (column 2 / index 1)."""
        out: dict[str, str] = dict.fromkeys(providers, "")
        if not providers or data is None:
            return out
        n = len(providers)

        if isinstance(data, pd.DataFrame):
            nrows = len(data.index)
            ncol = int(data.shape[1]) if len(data.shape) > 1 else 0
            key_col = 1 if ncol >= 2 else 0
            for i in range(min(n, nrows)):
                cell = data.iloc[i, key_col]
                if cell is None or (isinstance(cell, float) and pd.isna(cell)):
                    v = ""
                else:
                    v = str(cell).strip()
                    if v.lower() == "nan":
                        v = ""
                out[providers[i]] = v
            return out

        try:
            import numpy as np

            if isinstance(data, np.ndarray) and data.ndim == 2:
                ncol = int(data.shape[1])
                key_col = 1 if ncol >= 2 else 0
                for i in range(min(n, int(data.shape[0]))):
                    raw = data[i, key_col]
                    s = "" if raw is None else str(raw).strip()
                    if s.lower() == "nan":
                        s = ""
                    out[providers[i]] = s
                return out
        except ImportError:
            pass

        if isinstance(data, (list, tuple)):
            for i, prov in enumerate(providers):
                if i >= len(data):
                    break
                row = data[i]
                if isinstance(row, (list, tuple)) and len(row) >= 2:
                    cell = row[1]
                elif isinstance(row, (list, tuple)) and len(row) == 1:
                    cell = row[0]
                else:
                    cell = ""
                out[prov] = (str(cell) if cell is not None else "").strip()
            return out

        return out

    def _llm_keys_map_to_dataframe(self, keys_map: dict[str, str]) -> pd.DataFrame:
        if len(self._llm_keys_df_columns) >= 2:
            cols = self._llm_keys_df_columns
        else:
            cols = [
                self._get_translation("config_llm_provider_label"),
                self._get_translation("config_llm_api_key_label"),
            ]
        rows = [
            [p, (keys_map.get(p) or "").strip()]
            for p in (self._config_llm_providers or [])
        ]
        return pd.DataFrame(rows, columns=cols)

    def _empty_llm_keys_dataframe(self) -> pd.DataFrame:
        return self._llm_keys_map_to_dataframe({})

    # Event handlers
    def _save_to_state(
        self,
        url: str,
        username: str,
        password: str,
        llm_provider_override: str,
        llm_keys_df: Any,
        current_state: dict | None,
        request: gr.Request | None = None,
    ) -> dict:
        """Save configuration into browser state and update process env."""
        try:
            url = (url or "").strip()
            username = (username or "").strip()
            password = (password or "").strip()
            llm_provider_override = (llm_provider_override or "").strip()

            merged_base = dict(current_state) if isinstance(current_state, dict) else {}
            keys_map = ConfigTab._normalize_llm_provider_api_keys(merged_base)
            parsed = ConfigTab._parse_llm_keys_dataframe(
                llm_keys_df,
                self._config_llm_providers,
            )
            for p in self._config_llm_providers:
                keys_map[p] = parsed.get(p, "")

            llm_api_key_override = ""
            if llm_provider_override:
                llm_api_key_override = keys_map.get(llm_provider_override, "")

            # Prepare new state dict
            new_state = {
                "url": url,
                "username": username,
                "password": password,
                "llm_provider_override": llm_provider_override,
                "llm_api_key_override": llm_api_key_override,
                "llm_provider_api_keys": keys_map,
            }

            # Determine accurate session id
            session_id = self._resolve_session_id(request)

            if session_id:
                set_session_config(session_id, new_state)
                self._reinitialize_session_llm(session_id)
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
                message = (
                    self._get_translation("config_save_error") + "\n\n" + f"{str(e)}"
                )
                gr.Warning(message)
            return current_state or {}
        else:
            return new_state

    def _load_from_state(
        self, state: Any, request: gr.Request | None = None
    ) -> tuple[Any, Any, Any, Any, Any]:
        """Load values from browser state and update fields."""
        try:
            # Normalize state across gradio versions (may come as tuple or dict)
            if isinstance(state, tuple) and len(state) > 0:
                state = state[0]
            if not isinstance(state, dict):
                state = {}

            # If no stored config, return no-ops — don't clear initial values
            keys_precheck = ConfigTab._normalize_llm_provider_api_keys(state)
            any_stored_llm_keys = bool(
                any((vk or "").strip() for vk in keys_precheck.values())
            )
            has_saved_config = (
                any(
                    state.get(k)
                    for k in (
                        "url",
                        "username",
                        "password",
                        "llm_provider_override",
                        "llm_api_key_override",
                    )
                )
                or any_stored_llm_keys
            )
            if not has_saved_config:
                return (
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.skip(),
                )

            url = state.get("url", "") or ""
            login = state.get("username", "") or ""
            pwd = state.get("password", "") or ""
            llm_provider = (state.get("llm_provider_override", "") or "").strip()

            keys_map = ConfigTab._normalize_llm_provider_api_keys(state)
            df_llm = self._llm_keys_map_to_dataframe(keys_map)

            logging.getLogger(__name__).debug(
                "ConfigTab._load_from_state: url_present=%s user_len=%s pwd_len=%s "
                "provider=%s",
                bool(url),
                len(login),
                len(pwd),
                llm_provider,
            )

            # Also propagate BrowserState snapshot into per-session store for backend
            try:
                session_id = self._resolve_session_id(request)

                if session_id:
                    llm_api_key = (keys_map.get(llm_provider, "") or "").strip()
                    if not llm_api_key:
                        llm_api_key = (
                            state.get("llm_api_key_override", "") or ""
                        ).strip()

                    set_session_config(
                        session_id,
                        {
                            "url": url,
                            "username": login,
                            "password": pwd,
                            "llm_provider_override": llm_provider,
                            "llm_api_key_override": llm_api_key,
                        },
                    )
                    self._reinitialize_session_llm(session_id)
                    logging.getLogger(__name__).debug(
                        "🔄 ConfigTab.load -> propagated BrowserState to session=%s",
                        session_id,
                    )
            except Exception:
                logging.getLogger(__name__).debug(
                    "ConfigTab.load -> failed to propagate BrowserState "
                    "to session store",
                    exc_info=True,
                )

            with suppress(Exception):
                gr.Info(self._get_translation("config_load_success"))
            return (
                gr.update(value=url),
                gr.update(value=login),
                gr.update(value=pwd),
                gr.update(value=llm_provider),
                gr.update(value=df_llm),
            )
        except Exception as e:
            logging.getLogger(__name__).exception("Load from browser state failed")
            with suppress(Exception):
                message = (
                    self._get_translation("config_load_error") + "\n\n" + f"{str(e)}"
                )
                gr.Warning(message)
            return (
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.skip(),
            )

    # Removed .env loading: rely solely on browser state

    def _clear_browser_storage(
        self, state: Any
    ) -> tuple[dict, Any, Any, Any, Any, Any]:
        """Clear browser-persisted state and reset input fields."""
        try:
            new_state: dict = {
                "url": "",
                "username": "",
                "password": "",
                "llm_provider_override": "",
                "llm_api_key_override": "",
                "llm_provider_api_keys": {},
            }

            with suppress(Exception):
                gr.Info(self._get_translation("config_clear_success"))
            empty_df = self._empty_llm_keys_dataframe()
            return (
                new_state,
                gr.update(value=""),
                gr.update(value=""),
                gr.update(value=""),
                gr.update(value=""),
                gr.update(value=empty_df),
            )
        except Exception as e:
            logging.getLogger(__name__).exception("Clear browser storage failed")
            with suppress(Exception):
                base = self._get_translation("config_clear_error")
                details = f"{str(e)}"
                gr.Warning(base + "\n\n" + details)
            # Return original state if clear failed
            return (
                state or {},
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.skip(),
            )

    def set_main_app(self, main_app: Any) -> None:
        """Set reference to main app for future integration needs."""
        self.main_app = main_app

    def _reinitialize_session_llm(self, session_id: str) -> None:
        """Re-initialize session LLM instance to pick up updated config."""
        try:
            if (
                self.main_app
                and hasattr(self.main_app, "session_manager")
                and self.main_app.session_manager
            ):
                session_data = self.main_app.session_manager.get_session_data(
                    session_id
                )
                if session_data and hasattr(session_data, "_initialize_session_agent"):
                    session_data._initialize_session_agent()  # noqa: SLF001
                    logging.getLogger(__name__).debug(
                        "🔄 ConfigTab -> re-initialized session LLM for %s",
                        session_id,
                    )
        except Exception:
            logging.getLogger(__name__).debug(
                "ConfigTab -> failed to re-initialize session LLM",
                exc_info=True,
            )

    def _resolve_session_id(self, request: gr.Request | None) -> str | None:
        """Resolve session ID from request or main app fallback."""
        try:
            if request and hasattr(request, "session_hash") and request.session_hash:
                return f"gradio_{request.session_hash}"
        except Exception:
            logging.getLogger(__name__).debug(
                "_resolve_session_id: request.parse failed", exc_info=True
            )

        if (
            self.main_app
            and hasattr(self.main_app, "session_manager")
            and hasattr(self.main_app.session_manager, "get_last_active_session_id")
        ):
            return self.main_app.session_manager.get_last_active_session_id()  # type: ignore[attr-defined]

        return None

    def get_components(self) -> dict[str, Any]:
        """Return created components."""
        return self.components

    # No get_current_config needed; backend reads from SessionManager per session

    def _get_translation(self, key: str) -> str:
        """Get a translation for a specific key (consistent with other tabs)"""
        # Always use direct translation for now to avoid i18n metadata issues
        return get_translation_key(key, self.language)
