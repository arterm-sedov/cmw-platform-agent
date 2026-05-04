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

    @staticmethod
    def use_dotenv_for_platform() -> bool:
        """When true, platform URL/login/password come from .env, not the Config UI."""
        return os.environ.get("CMW_USE_DOTENV", "true").lower() in (
            "1",
            "true",
            "yes",
        )

    @staticmethod
    def _resolve_config_llm_provider_ids(main_app: Any) -> list[str]:
        """Provider ids for API-key rows — from ``LLMManager`` only (no static fallback)."""
        log = logging.getLogger(__name__)
        if main_app is None or not getattr(main_app, "llm_manager", None):
            log.warning(
                "ConfigTab: no LLM manager when building API key rows; "
                "provider list empty (check app wiring / init order)."
            )
            return []
        try:
            ids = main_app.llm_manager.get_available_providers()
        except Exception:
            log.exception(
                "ConfigTab: get_available_providers() failed; API key rows omitted."
            )
            return []
        if not ids:
            log.warning(
                "ConfigTab: get_available_providers() returned no providers "
                "(no API keys / env for any configured provider?)."
            )
            return []
        return sorted(ids)

    def __init__(
        self,
        event_handlers: dict[str, Callable],
        language: str = "en",
        i18n_instance: gr.I18n | None = None,
    ) -> None:
        self.event_handlers = event_handlers
        self.components: dict[str, Any] = {}
        self.main_app = None  # Reference to main app if needed later
        self.sidebar_instance: Any = None  # Shared Sidebar — LLM controls live here
        self.language = language
        # Filled in _create_config_interface; tests may assign before calling handlers.
        self._config_llm_providers: list[str] = []
        self._llm_provider_key_inputs: list[gr.Textbox] = []
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

        with gr.TabItem(
            self._get_translation("tab_config"),
            id="config",
            render_children=True,
        ) as tab:
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

        # BrowserState default MUST be all empty strings. Do NOT bake env defaults
        # here — demo.load() fires before onMount() reads localStorage, so it
        # receives the default value. If provider is pre-filled, has_any_value
        # becomes True and url/username/password get cleared.
        self.components["config_state"] = gr.BrowserState(
            {
                "url": "",
                "username": "",
                "password": "",
                "llm_provider_api_keys": {},
            },
            storage_key="cmw_config_v1",
        )

        # Textbox initial values — always empty, BrowserState handles persistence
        url_init = ""
        login_init = ""
        password_init = ""

        platform_from_dotenv = self.use_dotenv_for_platform()

        # Platform: two columns — help/title (left) and credentials (right), like Gradio row+cards.
        with gr.Row(equal_height=True):
            with gr.Column(scale=1, elem_classes=["model-card"]):
                with gr.Row():
                    gr.Markdown(
                        f"### {self._get_translation('config_title')}",
                        elem_classes=["llm-selection-title"],
                    )
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1, min_width="50%"):
                        if platform_from_dotenv:
                            gr.Markdown(self._get_translation("config_platform_dotenv_notice"))
                        else:
                            gr.Markdown(self._get_translation("config_help"))

                    with gr.Column(
                        scale=1,
                        min_width="50%",
                        visible=not platform_from_dotenv,
                    ):

                        self.components["platform_url"] = gr.Textbox(
                            label=self._get_translation("config_platform_url"),
                            placeholder="https://your-comindware-host",
                            value=url_init,
                            lines=1,
                            max_lines=1,
                            scale=4,
                        )
                        self.components["username"] = gr.Textbox(
                            label=self._get_translation("config_username"),
                            value=login_init,
                            lines=1,
                            max_lines=1,
                            scale=1,
                        )
                        self.components["password"] = gr.Textbox(
                            label=self._get_translation("config_password"),
                            type="password",
                            value=password_init,
                            lines=1,
                            max_lines=1,
                            scale=1,
                        )
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                if self.sidebar_instance is not None:
                    self.sidebar_instance.mount_llm_selection_ui()
                else:
                    gr.Markdown("*LLM selection unavailable (internal).*")

            with gr.Column(scale=1, elem_classes=["model-card"]):
                self._config_llm_providers = self._resolve_config_llm_provider_ids(
                    self.main_app
                )
                _api_keys_heading = self._get_translation(
                    "config_llm_api_keys_table_label"
                )
                gr.Markdown(
                    f"### {_api_keys_heading}",
                    elem_classes=["llm-selection-title"],
                )
                self._llm_provider_key_inputs = []
                if not self._config_llm_providers:
                    gr.Markdown(
                        self._get_translation("config_llm_providers_none_message")
                    )
                else:
                    with gr.Column():
                        for prov in self._config_llm_providers:
                            tb = gr.Textbox(
                                label=prov,
                                type="password",
                                value="",
                                lines=1,
                                max_lines=1,
                                placeholder="sk-...",
                            )
                            self._llm_provider_key_inputs.append(tb)
                    gr.Markdown(
                        "*"
                        + self._get_translation("config_llm_empty_means_default")
                        + "*"
                    )
                self.components["llm_provider_key_inputs"] = (
                    self._llm_provider_key_inputs
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

    @staticmethod
    def _platform_credentials_from_env() -> tuple[str, str, str]:
        """CMW_BASE_URL, CMW_LOGIN, CMW_PASSWORD from process environment."""
        return (
            (os.environ.get("CMW_BASE_URL") or "").strip(),
            (os.environ.get("CMW_LOGIN") or "").strip(),
            (os.environ.get("CMW_PASSWORD") or "").strip(),
        )

    def _session_config_payload(
        self, browser_snapshot: dict[str, Any]
    ) -> dict[str, Any]:
        """Platform URL/login/password from .env if dotenv mode; else from snapshot."""
        out = dict(browser_snapshot) if isinstance(browser_snapshot, dict) else {}
        keys_raw = out.get("llm_provider_api_keys")
        keys_map: dict[str, str] = {}
        if isinstance(keys_raw, dict):
            keys_map = {
                str(k).strip(): (str(v).strip() if v is not None else "")
                for k, v in keys_raw.items()
                if str(k).strip()
            }
        if self.use_dotenv_for_platform():
            u, login, pwd = self._platform_credentials_from_env()
            out["url"] = u
            out["username"] = login
            out["password"] = pwd
        out["llm_provider_api_keys"] = keys_map
        return out

    def _browser_state_public_snapshot(
        self, merged_state: dict[str, Any]
    ) -> dict[str, Any]:
        """Do not persist .env platform secrets into BrowserState when using dotenv."""
        if not isinstance(merged_state, dict):
            return {}
        if not self.use_dotenv_for_platform():
            return dict(merged_state)
        pub = dict(merged_state)
        pub["url"] = ""
        pub["username"] = ""
        pub["password"] = ""
        return pub

    @staticmethod
    def _tuple_empty_textbox_updates(count: int) -> tuple[Any, ...]:
        """One fresh ``gr.update(value=\"\")`` per textbox (tuple multiply reuses one object)."""
        return tuple(gr.update(value="") for _ in range(max(0, count)))

    @staticmethod
    def _tuple_skip_updates(count: int) -> tuple[Any, ...]:
        return tuple(gr.skip() for _ in range(max(0, count)))

    @staticmethod
    def _tuple_noop_updates(count: int) -> tuple[Any, ...]:
        return tuple(gr.update() for _ in range(max(0, count)))

    def _connect_events(self) -> None:
        """Wire events for Save/Load configuration."""
        logging.getLogger(__name__).debug("🔗 ConfigTab: Connecting events...")

        # Save to browser state (and update runtime env)
        llm_extra_inputs: list[Any] = []
        sb = self.sidebar_instance
        if sb is not None and getattr(sb, "components", None):
            for key in (
                "provider_model_selector",
                "use_fallback_model",
                "fallback_model_selector",
                "compression_enabled",
            ):
                comp = sb.components.get(key)
                if comp is not None:
                    llm_extra_inputs.append(comp)

        save_outputs: list[Any] = [self.components["config_state"]]
        save_outputs.extend(llm_extra_inputs)

        self.components["save_btn"].click(
            fn=self._save_to_state,
            inputs=[
                self.components["platform_url"],
                self.components["username"],
                self.components["password"],
                *self._llm_provider_key_inputs,
                *llm_extra_inputs,
                self.components["config_state"],
            ],
            outputs=save_outputs,
        )

        # Load from browser state
        load_outputs: list[Any] = [
            self.components["platform_url"],
            self.components["username"],
            self.components["password"],
            *self._llm_provider_key_inputs,
        ]
        load_outputs.extend(llm_extra_inputs)

        self.components["load_btn"].click(
            fn=self._load_from_state,
            inputs=[self.components["config_state"]],
            outputs=load_outputs,
        )

        # Clear browser storage and reset fields
        clear_outputs: list[Any] = [
            self.components["config_state"],
            self.components["platform_url"],
            self.components["username"],
            self.components["password"],
            *self._llm_provider_key_inputs,
        ]
        clear_outputs.extend(llm_extra_inputs)

        self.components["clear_storage_btn"].click(
            fn=self._clear_browser_storage,
            inputs=[self.components["config_state"]],
            outputs=clear_outputs,
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
        return mp

    @staticmethod
    def _key_values_to_map(values: list[Any], providers: list[str]) -> dict[str, str]:
        """Align password textbox values with the fixed provider order."""
        out: dict[str, str] = dict.fromkeys(providers, "")
        if not providers:
            return out
        for i, p in enumerate(providers):
            if i >= len(values):
                break
            v = values[i]
            out[p] = (str(v) if v is not None else "").strip()
        return out

    @staticmethod
    def _parse_llm_keys_dataframe(
        data: Any,
        providers: list[str],
    ) -> dict[str, str]:
        """Best-effort parse of legacy Dataframe / 2D payloads (e.g. pasted state)."""
        if data is None:
            return dict.fromkeys(providers, "")
        if (
            isinstance(data, (list, tuple))
            and data
            and isinstance(
                data[0],
                (str, int, float),
            )
        ):
            flat = [str(x).strip() if x is not None else "" for x in data]
            return ConfigTab._key_values_to_map(flat, providers)
        out: dict[str, str] = dict.fromkeys(providers, "")
        if not providers:
            return out
        n = len(providers)

        try:
            import pandas as pd

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
        except ImportError:
            pass

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

    def _save_to_state(
        self,
        url: str,
        username: str,
        password: str,
        request: gr.Request | None = None,
        *rest: Any,
    ) -> tuple[Any, ...]:
        """Save configuration into browser state and update process env.

        ``request`` must come **before** ``*rest``: Gradio only injects ``gr.Request``
        for positional-or-keyword parameters listed before ``*args`` (see
        ``gradio.helpers.special_args``).

        Returns ``(browser_state_dict, *gr.update()``...)`` when Save wires LLM
        components as passthrough outputs so their values are not cleared.
        """
        prior_browser: dict[Any, Any] = {}
        if rest and isinstance(rest[-1], dict):
            prior_browser = dict(rest[-1])

        try:
            if len(rest) < 1:
                logging.getLogger(__name__).warning(
                    "ConfigTab._save_to_state: missing trailing BrowserState tuple"
                )
                n_llm_side = max(0, len(rest) - 1 - len(self._config_llm_providers))
                return (prior_browser, *tuple(gr.update() for _ in range(n_llm_side)))

            url = (url or "").strip()
            username = (username or "").strip()
            password = (password or "").strip()
            current_state_raw = rest[-1]
            key_inputs = rest[:-1]
            n_llm_side = max(0, len(rest) - 1 - len(self._config_llm_providers))
            passthrough = tuple(gr.update() for _ in range(n_llm_side))
            key_inputs = key_inputs[: len(key_inputs) - n_llm_side]

            merged_base = (
                dict(current_state_raw) if isinstance(current_state_raw, dict) else {}
            )
            keys_map = ConfigTab._normalize_llm_provider_api_keys(merged_base)
            if len(key_inputs) == len(self._config_llm_providers):
                parsed = ConfigTab._key_values_to_map(
                    list(key_inputs), self._config_llm_providers
                )
            else:
                logging.getLogger(__name__).debug(
                    "ConfigTab.save: key input count %s != providers %s; "
                    "falling back to dataframe-style parse",
                    len(key_inputs),
                    len(self._config_llm_providers),
                )
                parsed = ConfigTab._parse_llm_keys_dataframe(
                    key_inputs[0] if len(key_inputs) == 1 else key_inputs,
                    self._config_llm_providers,
                )
            for p in self._config_llm_providers:
                keys_map[p] = parsed.get(p, "")

            # Prepare new state dict (browser row); platform triple may be replaced
            # from .env for the live session when ``CMW_USE_DOTENV`` is enabled.
            new_state = {
                "url": url,
                "username": username,
                "password": password,
                "llm_provider_api_keys": keys_map,
            }
            session_payload = self._session_config_payload(new_state)
            stored_browser = self._browser_state_public_snapshot(session_payload)

            session_id = self._resolve_session_id(request)

            if session_id:
                self._apply_session_config(session_id, session_payload)
                try:
                    su = (session_payload.get("url") or "").strip()
                    sl = (session_payload.get("username") or "").strip()
                    sp = (session_payload.get("password") or "").strip()
                    masked = {
                        "url_present": bool(su),
                        "username_len": len(sl),
                        "password_len": len(sp),
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
            elif any((vk or "").strip() for vk in keys_map.values()):
                logging.getLogger(__name__).warning(
                    "ConfigTab.save: no session id; LLM API keys were not applied "
                    "to the server session (browser storage still updated)",
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
            return (prior_browser, *passthrough)
        else:
            return (stored_browser, *passthrough)

    def _load_from_state(
        self, state: Any, request: gr.Request | None = None
    ) -> tuple[Any, ...]:
        """Load values from browser state and update fields."""
        n_keys = len(self._config_llm_providers)
        n_llm_side = 0
        sb = self.sidebar_instance
        if sb is not None and getattr(sb, "components", None):
            for key in (
                "provider_model_selector",
                "use_fallback_model",
                "fallback_model_selector",
                "compression_enabled",
            ):
                if sb.components.get(key) is not None:
                    n_llm_side += 1
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
                any(state.get(k) for k in ("url", "username", "password"))
                or any_stored_llm_keys
            )
            if self.use_dotenv_for_platform():
                has_saved_config = True
            if not has_saved_config:
                return (
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    *ConfigTab._tuple_skip_updates(n_keys),
                    *ConfigTab._tuple_noop_updates(n_llm_side),
                )

            url = state.get("url", "") or ""
            login = state.get("username", "") or ""
            pwd = state.get("password", "") or ""

            keys_map = ConfigTab._normalize_llm_provider_api_keys(state)

            browser_row = {
                "url": url,
                "username": login,
                "password": pwd,
                "llm_provider_api_keys": keys_map,
            }
            payload = self._session_config_payload(browser_row)
            su = (payload.get("url") or "").strip()
            sl = (payload.get("username") or "").strip()
            sp = (payload.get("password") or "").strip()

            logging.getLogger(__name__).debug(
                "ConfigTab._load_from_state: url_present=%s user_len=%s pwd_len=%s",
                bool(su),
                len(sl),
                len(sp),
            )

            session_id = self._resolve_session_id(request)
            try:
                if session_id:
                    self._apply_session_config(session_id, payload)
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
            if self.use_dotenv_for_platform():
                platform_updates = (gr.update(), gr.update(), gr.update())
            else:
                platform_updates = (
                    gr.update(value=url),
                    gr.update(value=login),
                    gr.update(value=pwd),
                )
            return (
                *platform_updates,
                *(
                    gr.update(value=(keys_map.get(p) or "").strip())
                    for p in self._config_llm_providers
                ),
                *ConfigTab._tuple_noop_updates(n_llm_side),
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
                *ConfigTab._tuple_skip_updates(n_keys),
                *ConfigTab._tuple_noop_updates(n_llm_side),
            )

    # Removed .env loading: rely solely on browser state

    def _clear_browser_storage(self, state: Any) -> tuple[Any, ...]:
        """Clear browser-persisted state and reset input fields."""
        n_key = len(self._llm_provider_key_inputs) or len(self._config_llm_providers)
        n_llm_side = 0
        sb = self.sidebar_instance
        if sb is not None and getattr(sb, "components", None):
            for key in (
                "provider_model_selector",
                "use_fallback_model",
                "fallback_model_selector",
                "compression_enabled",
            ):
                if sb.components.get(key) is not None:
                    n_llm_side += 1
        try:
            new_state: dict[str, Any] = {
                "url": "",
                "username": "",
                "password": "",
                "llm_provider_api_keys": {},
            }
            session_id = self._resolve_session_id(None)
            if session_id:
                session_payload = self._session_config_payload(new_state)
                self._apply_session_config(session_id, session_payload)
                stored_browser = self._browser_state_public_snapshot(session_payload)
            else:
                stored_browser = new_state

            with suppress(Exception):
                gr.Info(self._get_translation("config_clear_success"))
            return (
                stored_browser,
                gr.update(value=""),
                gr.update(value=""),
                gr.update(value=""),
                *ConfigTab._tuple_empty_textbox_updates(n_key),
                *ConfigTab._tuple_noop_updates(n_llm_side),
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
                *ConfigTab._tuple_skip_updates(n_key),
                *ConfigTab._tuple_noop_updates(n_llm_side),
            )

    def set_main_app(self, main_app: Any) -> None:
        """Set reference to main app for future integration needs."""
        self.main_app = main_app

    def set_sidebar_instance(self, sidebar: Any) -> None:
        """Share Sidebar instance so LLM controls render inside Config."""
        self.sidebar_instance = sidebar

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

    def _apply_session_config(self, session_id: str, state: dict[str, Any]) -> None:
        """Push config snapshot to server session store and rebuild LLM for keys."""
        set_session_config(session_id, state)
        self._reinitialize_session_llm(session_id)

    def _resolve_session_id(self, request: gr.Request | None) -> str | None:
        """Resolve session ID — match chat/sidebar via SessionManager.get_session_id."""
        try:
            sm = (
                getattr(self.main_app, "session_manager", None)
                if self.main_app
                else None
            )
            if sm is not None and request is not None:
                return sm.get_session_id(request)
        except Exception:
            logging.getLogger(__name__).debug(
                "_resolve_session_id: get_session_id failed", exc_info=True
            )

        if (
            self.main_app
            and getattr(self.main_app, "session_manager", None) is not None
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
