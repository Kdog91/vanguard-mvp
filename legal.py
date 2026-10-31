"""Privacy policy and terms of use, shown at ?page=privacy and ?page=terms."""
import streamlit as st

UPDATED = "October 30, 2026"
CONTACT = "vanguarddataanalytics@gmail.com"
COMPANY = "Vanguard Data Analytics LLC, 1118 Lincoln Ave, Rockford, IL 61102"

PRIVACY = f"""
# Privacy Policy

*Last updated: {UPDATED}*

Vanguard Analytics ("the app") is operated by {COMPANY} ("we", "us"). This policy explains what we collect when
you use the app and what we do with it.

## What we collect
- **Sign-in information.** Sign-in is handled by Auth0 (Okta). When you sign up with an email and password, or
  with an account such as Google, the app receives your **name and email address**. We use them only to identify
  your account, show who is signed in, and control access. We never receive or store your password.
- **Data you load.** Files you upload, web addresses you load, and database tables you connect are processed so the
  app can clean, analyze and chart them. Unless you choose to save a project, this data is kept only in memory for
  your session and is discarded when the session ends.
- **Saved projects (only if you click Save).** The data you loaded, your cleaning choices and the analysis you
  picked are stored so you can reopen them. Each project is encrypted by the app before it is stored, with a key
  our storage provider does not have, and is linked to your account only (by a one-way code, not your email).
  Columns that look like names, emails, phone or ID numbers are replaced with codes before saving unless you turn
  that off. You can delete one project or all of them at any time from **My projects**; a project you have not
  opened for 180 days is deleted automatically. You can save up to 25 projects.
- **Security records.** The app records security events: sign-in and sign-out, the type and size of files you
  load, the host name of web addresses you load, downloads, and errors. Your account appears in these records as
  a one-way code, not your email. The records never contain your data, file contents or passwords.
- **Technical information.** Our hosting provider (Streamlit Community Cloud, operated by Snowflake Inc.) may record
  standard technical information such as IP address, browser type and usage logs to run and secure the service.

## What we do not do
- We do **not** sell your personal information or your data.
- We do **not** use your data to train models for anyone else, or share it with other users.
- We do **not** use your sign-in account for anything other than signing in.

## Who can see your data
Only you, while signed in. Saved projects are encrypted and stored under your account's code, and no other
account can list or open them. We may access stored data only when needed to provide support you request, keep the
service secure, or comply with the law. Database passwords you enter are used only to connect and are not saved.

## Third-party services
- **Auth0 by Okta** (sign-in): see okta.com/privacy-policy. If you sign in with Google, Google's Privacy Policy
  (policies.google.com/privacy) also applies.
- **Streamlit Community Cloud / Snowflake** (hosting).
- **Supabase** (storage for saved projects, in the United States; data is encrypted before it reaches Supabase).
- **USAspending.gov** (public federal award data, when you choose to load it).

## Your choices
You can use the sample data without signing in, sign out at any time, delete saved projects, and ask us to delete
your account information by emailing {CONTACT}. If you signed in with Google, you can also remove the app's access
in your Google Account settings under *Security → Third-party connections*.

## Private details in downloads
When the app finds columns that look like names, email addresses, phone numbers or ID numbers, it replaces them
with codes in downloads and Word reports by default. You can turn this off for a download you are entitled to.

## Sensitive data
Do not upload protected health information (PHI), classified or controlled unclassified information (CUI), payment
card numbers, Social Security numbers, or other data you are not permitted to share with a cloud service. For
sensitive work, we can run the app inside your own environment.

## If something goes wrong
If we learn of a security incident affecting your information, we will notify you as required by Illinois law
(the Personal Information Protection Act) and other laws that apply. Report a suspected problem to {CONTACT}.

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
   the service, probe or scan networks through it, or access other people's accounts or data.
3. **Data not allowed on the public app.** Do not load protected health information (PHI) covered by HIPAA,
   controlled unclassified information (CUI) or classified information, payment card numbers, Social Security
   numbers, or any data your agreements or the law do not allow you to place on a cloud service. If you need to
   analyze that kind of data, contact us about a copy installed inside your own environment.
4. **Results are decision support.** The app produces statistics, models and forecasts that can be wrong. Review
   results with someone who knows the subject before relying on them. The app is not legal, financial or medical advice.
5. **No guarantee.** The app is provided "as is" and may change or be unavailable at times. To the fullest extent the
   law allows, we are not liable for indirect or consequential losses arising from its use.
6. **Accounts.** We may suspend access that breaks these terms. You can stop using the app at any time.
7. **Changes.** We may update these terms and will post the new date here.
8. **Governing law.** Illinois, United States.

Questions: **{CONTACT}**.
"""


def route() -> bool:
    """Show a legal page if the address asks for one. Returns True when a page was shown."""
    page = st.query_params.get("page", "")
    if page == "authcheck":
        import auth
        if auth.configured() and not auth.is_admin():
            st.title("Sign-in check")
            st.info("This page is for the app's administrator. Sign in with an admin account "
                    "(listed under [access] admin_emails in Secrets) to see it.")
            if not auth.logged_in():
                st.button("Sign in", on_click=st.login, type="primary")
            st.link_button("Back to Vanguard Analytics", "./")
            return True
        auth.render_check()
        import projects_ui
        projects_ui.render_admin_check()
        return True
    if page not in ("privacy", "terms"):
        return False
    st.markdown(PRIVACY if page == "privacy" else TERMS)
    st.divider()
    st.link_button("Back to Vanguard Analytics", "./")
    return True


def footer_links():
    st.caption("[Privacy Policy](?page=privacy) · [Terms of Use](?page=terms) · "
               f"[{CONTACT}](mailto:{CONTACT})")
