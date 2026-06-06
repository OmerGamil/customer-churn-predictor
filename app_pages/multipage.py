"""
multipage.py — Streamlit multipage navigation helper.

Provides the MultiPage class used to register and render
multiple dashboard pages via a sidebar radio selector.
"""

import streamlit as st


class MultiPage:
    """Manages multiple Streamlit app pages via sidebar navigation."""

    def __init__(self, app_name) -> None:
        """
        Initialise the app with a name and an empty page list.

        Args:
            app_name (str): Title shown in the browser tab and page header.
        """
        self.pages = []
        self.app_name = app_name
        st.set_page_config(
            page_title=self.app_name,
            page_icon="📊"
        )

    def add_page(self, title, func) -> None:
        """
        Register a new page.

        Args:
            title (str): Label shown in the sidebar navigation.
            func (callable): Function that renders the page content.
        """
        self.pages.append({"title": title, "function": func})

    def run(self):
        """Render the sidebar and call the selected page function."""
        st.title(self.app_name)
        page = st.sidebar.radio(
            "Navigation",
            self.pages,
            format_func=lambda page: page["title"]
        )
        # Call the render function for the selected page
        page["function"]()
