import streamlit as st

# Set the page configuration
st.set_page_config(page_title="Payment Portal", page_icon="💳")

# Main Title
st.title("⚠️ Payment Required")

# Subheader or Message
st.header("Please pay your outstanding balance to continue.")

# Displaying an amount (Optional)
st.metric(label="Total Due", value="$50.00")

# A separator
st.divider()

# A Payment Button
if st.button("Pay Now"):
    # What happens when the user clicks the button
    st.success("Payment Received! Thank you.")
    st.balloons()
else:
    st.warning("Status: Pending")
