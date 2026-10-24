"""Privacy policy and terms of use, shown at ?page=privacy and ?page=terms (links for Google sign-in branding)."""
import streamlit as st

UPDATED = "October 24, 2026"
CONTACT = "vanguarddataanalytics@gmail.com"
COMPANY = "Vanguard Data Analytics LLC, 1118 Lincoln Ave, Rockford, IL 61102"

PRIVACY = f"""
# Privacy Policy

*Last updated: {UPDATED}*

Vanguard Analytics ("the app") is operated by {COMPANY} ("we", "us"). This policy explains what we collect when
you use the app and what we do with it.

## What we collect
- **Sign-in information.** When you sign in with Google, Google shares your **name and email address** with the app.
  We use them only to identify your account, show who is signed in, and control access. We do not receive your
  Google password.
- **Data you load.** Files you upload, web addresses you load, and database tables you connect are processed so the
  app can clean, analyze and chart them. Unless you choose to save a project, this data is kept only in memory for
  your session and is discarded when the session ends.
- **Saved projects (if you use them).** If you save a project, its settings and data are stored so you can reopen it.
  You can delete saved projects at any time.
- **Technical information.** Our hosting provider (Streamlit Community Cloud, operated by Snowflake Inc.) may record
  standard technical information such as IP address, browser type and usage logs to run and secure the service.

## What we do not do
- We do **not** sell your personal information or your data.
- We do **not** use your data to train models for anyone else, or share it with other users.
- We do **not** use your Google account for anything other than signing in.

## Who can see your data
Only you, while signed in. We may access stored data only when needed to provide support you request, keep the
service secure, or comply with the law. Database passwords you enter are used only to connect and are not saved.

## Third-party services
- **Google** (sign-in): see Google's Privacy Policy at policies.google.com/privacy.
- **Streamlit Community Cloud / Snowflake** (hosting).
- **USAspending.gov** (public federal award data, when you choose to load it).

## Your choices
You can use the sample data without signing in, sign out at any time, delete saved projects, and ask us to delete
your account information by emailing {CONTACT}. You can also remove the app's access in your Google Account settings
under *Security → Third-party connections*.

## Sensitive data
Do not upload protected health information, classified or controlled unclassified information, or other data you
are not permitted to share with a cloud service. For sensitive work, we can run the app inside your own environment.

## Children
The app is not intended for anyone under 18.

## Changes and contact
We will post any changes on this page with a new date. Questions: **{CONTACT}**.
"""

TERMS = f"""
# Terms of Use

*Last updated: {UPDATED}*

These terms apply to Vanguard Analytics ("the app"), provided by {COMPANY}. By using the app you agree to them.

1. **Your data stays yours.** You keep all rights to the data you load and the results you create. You give us
   permission to process it only to run the app for you.
2. **Use it lawfully.** Only load data you have the right to use. Do not upload malware, attempt to break or overload
   the service, or access other people's accounts or data.
3. **Results are decision support.** The app produces statistics, models and forecasts that can be wrong. Review
   results with someone who knows the subject before relying on them. The app is not legal, financial or medical advice.
4. **No guarantee.** The app is provided "as is" and may change or be unavailable at times. To the fullest extent the
   law allows, we are not liable for indirect or consequential losses arising from its use.
5. **Accounts.** We may suspend access that breaks these terms. You can stop using the app at any time.
6. **Changes.** We may update these terms and will post the new date here.
7. **Governing law.** Illinois, United States.

Questions: **{CONTACT}**.
"""


def route() -> bool:
    """Show a legal page if the address asks for one. Returns True when a page was shown."""
    page = st.query_params.get("page", "")
    if page not in ("privacy", "terms"):
        return False
    st.markdown(PRIVACY if page == "privacy" else TERMS)
    st.divider()
    st.link_button("Back to Vanguard Analytics", "./")
    return True


def footer_links():
    st.caption("[Privacy Policy](?page=privacy) · [Terms of Use](?page=terms) · "
               f"[{CONTACT}](mailto:{CONTACT})")
