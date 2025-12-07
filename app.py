# ======================== TAB 2: METRICS ========================
with tab2:
    st.markdown("### Model Evaluation Metrics (on Test Set)")
    st.write(
        "These performance visualizations are based on the model's validation/testing results. "
        "They help determine how reliably the system can classify scrap metal versus trash when used "
        "by users in Orlan’s Junkshop."
    )
    
    # ---------------------------------------------------------
    # Confusion Matrix (Row 1)
    # ---------------------------------------------------------
    st.image("confusion_matrix.png", caption="Confusion Matrix")
    st.markdown(
        """
        **Confusion Matrix Interpretation**  
        • The model correctly identifies **912 metal items** and **236 trash items**, showing strong reliability.  
        • A small number of background areas are misclassified as scrap (**67 cases**), but this does not heavily affect performance.  
        • Mislabeling between metal and trash is **very low** (only **5 metal → trash** and **11 trash → metal**).  
        
        📌 This means the Scrap Checker is highly capable of preventing recyclable metal from being misclassified as trash — leading to reduced losses and improved recycling efficiency.
        """
    )
    
    st.markdown("---")

    # ---------------------------------------------------------
    # F1 Confidence Curve (Row 2)
    # ---------------------------------------------------------
    st.image("f1_confidence_curve.png", caption="F1-Confidence Curve")
    st.markdown(
        """
        **F1-Confidence Curve Interpretation**  
        • The peak performance occurs around **0.455 confidence threshold**, where F1 ≈ **0.93**.  
        • The curve remains high and stable across a wide confidence range (~0.3–0.85), showing excellent **balance between precision and recall**.  
        • This confirms that the model consistently performs well even if object shapes, lighting, or camera angles vary.  
        
        📌 For practical use, a default confidence of **0.50** ensures accurate detection while still finding metals that are partially covered, dirty, or deformed.
        """
    )
