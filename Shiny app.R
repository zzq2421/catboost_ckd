# 1. Custom SHAP Visualization Function (Adapted for CatBoost)####

library(shiny)         # For building interactive web apps
library(catboost)      # For loading CatBoost models and predictions
library(ggplot2)       # For data visualization
library(shapviz)       # For SHAP value visualization
library(kernelshap)    # For SHAP value calculation (kernel method)
library(bslib)         # For custom themes and styles

# Load pre-trained model (core file)
# setwd("Path of the model")
cat_final <- readRDS("CKD prediction model.rds")

shapviz.catboost.Model <- function(object, X_pred, X = X_pred, collapse = NULL, ...) {
  if (!requireNamespace("catboost", quietly = TRUE)) {
    stop("Package 'catboost' is not installed. Please install it first.")
  }
  stopifnot(
    "X must be a matrix or data frame, not a catboost.Pool object" =
      is.matrix(X) || is.data.frame(X),
    "X_pred must be a matrix, data frame, or catboost.Pool object" =
      is.matrix(X_pred) || is.data.frame(X_pred) || inherits(X_pred, "catboost.Pool"),
    "X_pred must contain column names" = !is.null(colnames(X_pred))
  )
  
  if (!inherits(X_pred, "catboost.Pool")) {
    X_pred <- catboost.load_pool(X_pred)
  }
  
  S <- catboost.get_feature_importance(object, X_pred, type = "ShapValues", ...)
  pp <- ncol(X_pred) + 1L          
  baseline <- S[1L, pp]            
  S <- S[, -pp, drop = FALSE]      
  colnames(S) <- colnames(X_pred)  
  shapviz(S, X = X, baseline = baseline, collapse = collapse)
}

# 2. Define User Interface (UI)####
ui <- fluidPage(
  # Global style settings
  theme = bslib::bs_theme(
    bg = "white", 
    fg = "#333333", 
    primary = "#2c3e50",
    base_font = font_google("Lato")  # Enhance font aesthetics with Google Fonts
  ),
  tags$head(
    tags$style(HTML("
      /* Title style */
      .main-title {
        color: #2c3e50;
        font-weight: bold;
        text-align: center;
        margin: 20px 0 30px;
        font-size: 28px;
        text-shadow: 1px 1px 3px rgba(255,255,255,0.8);
      }
      /* Description box style */
      .description-box {
        background-color: #e8f4f8;
        padding: 15px 20px;
        border-radius: 8px;
        margin-bottom: 25px;
        border-left: 4px solid #3498db;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
      }
      .description-title {
        color: #2980b9;
        font-weight: bold;
        margin-top: 0;
        margin-bottom: 10px;
      }
      .description-text {
        color: #34495e;
        line-height: 1.6;
      }
      /* Input panel style */
      .sidebar-panel {
        background-color: #ffffea;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        padding: 20px;
      }
      /* Button style */
      .prediction-btn {
        background-color: #3498db;
        color: white;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
        font-weight: bold;
        border-radius: 5px;
        width: 100%;
        margin-top: 15px;
        transition: background-color 0.3s;
      }
      .prediction-btn:hover {
        background-color: #2980b9;
      }
      /* Results panel style */
      .results-panel {
        margin-top: 20px;
      }
      /* Prediction results style */
      .prediction-results {
        background-color: #fbe0e0;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        padding: 25px;
        margin-bottom: 20px;
      }
      /* SHAP plot area style */
      .shap-plot {
        background-color: #fff0d4;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        padding: 25px;
      }
      .result-title {
        color: #2c3e50;
        border-bottom: 2px solid #ecf0f1;
        padding-bottom: 10px;
        margin-bottom: 20px;
      }
      /* Input label style */
      .shiny-input-container > label {
        color: #2c3e50;
        font-weight: 600;
        margin-bottom: 5px;
        display: block;
      }
      /* Prediction result bold style */
      .pred-result strong {
        color: #2980b9;
      }
    "))
  ),
  
  # App title
  tags$h1(class = "main-title", "Predicting CKD in Community-Dwelling Older Adults with High Glycemic Status"),
  
  # Calculator description (English)
  div(class = "description-box",
      tags$h3(class = "description-title", "About This Calculator"),
      p(class = "description-text", 
        "This calculator uses the CatBoost machine learning model to assess the risk of chronic kidney disease (CKD) based on eight key indicators, trained on extensive patient data."),
      p(class = "description-text", 
        "Note: For reference only. Please consult a healthcare professional for diagnosis and treatment.")
  ),
  
  # Sidebar layout (input on left, output on right)
  sidebarLayout(
    # Sidebar panel (input area)
    sidebarPanel(class = "sidebar-panel",
                 # Input fields (variable name + unit)
                 numericInput("Age", "Age (years)", value = 60, min = 1, max = 120),
                 numericInput("SBP", "Systolic Blood Pressure (mmHg)", value = 120, min = 50, max = 250),
                 numericInput("HbA1c", "Glycated Hemoglobin (%)", value = 6.5, min = 3, max = 20),
                 numericInput("Scr", "Serum Creatinine (μmol/L)", value = 65, min = 20, max = 200),
                 numericInput("UA", "Uric Acid (μmol/L)", value = 327, min = 100, max = 1000),
                 numericInput("Hb", "Hemoglobin (g/L)", value = 140, min = 50, max = 250),
                 numericInput("TG", "Triglycerides (mmol/L)", value = 1.5, min = 0, max = 20),
                 numericInput("BUN", "Blood Urea Nitrogen (mmol/L)", value = 5.5, min = 0, max = 50),
                 
                 # Prediction button (triggers prediction on click)
                 actionButton("predictBtn", "Predict CKD Risk", class = "prediction-btn")
    ),
    
    # Main panel (output area)
    mainPanel(class = "results-panel",
              # Prediction results display area
              div(class = "prediction-results",
                  h3(class = "result-title", "Prediction Results (Threshold: 25%)"),
                  htmlOutput("predictionResult")
              ),
              
              # SHAP plot display area
              div(class = "shap-plot",
                  h3(class = "result-title", "SHAP Feature Contribution Plot"),
                  plotOutput("shapForcePlot", height = "400px")
              )
    )
  )
)

# 3. Define Server Logic (Backend Calculation)####
server <- function(input, output) {
  # Define prediction function
  predictCKD <- function() {
    new_data <- data.frame(
      Age = input$Age,
      SBP = input$SBP,
      HbA1c = input$HbA1c,
      Scr = input$Scr,
      UA = input$UA,
      Hb = input$Hb,
      TG = input$TG,
      BUN = input$BUN
    )
    
    new_pool <- catboost.load_pool(data = as.matrix(new_data), label = NULL)
    pred_prob <- catboost.predict(cat_final, pool = new_pool, prediction_type = "Probability")
    pred_label <- ifelse(pred_prob > 0.2, "CKD Positive", "CKD Negative")
    
    return(list(
      probability = pred_prob,
      label = pred_label,
      data = new_data
    ))
  }
  
  # Listen for prediction button clicks
  observeEvent(input$predictBtn, {
    prediction <- predictCKD()
    
    # Output prediction results (HTML format)
    output$predictionResult <- renderUI({
      HTML(
        paste0(
          "CKD Prediction Probability: <strong>", round(prediction$probability * 100, 2), "%</strong><br>",
          "Prediction Result: <strong>", prediction$label, "</strong>"
        )
      )
    })
    
    # Generate and output SHAP feature contribution plot
    output$shapForcePlot <- renderPlot({
      shp <- shapviz.catboost.Model(cat_final, X_pred = prediction$data)
      
      p_lt <- sv_force(shp, row_id = 1) +
        theme_minimal() +  # Use cleaner theme
        scale_fill_manual(values = c("#ff8696", "#bbe6dd")) +  # More distinct contrast colors
        labs(title = "Feature Contribution to CKD Risk Prediction") +
        theme(
          plot.title = element_text(hjust = 0.5, size = 14, face = "bold", color = "#2c3e50"),
          axis.title = element_text(face = "bold", size = 11, color = "#34495e"),
          plot.margin = unit(c(0.8, 0.8, 0.8, 0.8), "cm"),
          panel.grid.major = element_line(linewidth = 0.5, color = "#ecf0f1"),
          legend.position = "none"  # Remove legend for cleaner look
        )
      
      print(p_lt)
    })
  })
}

# 4. Run the Shiny App ####
shinyApp(ui = ui, server = server)

# 5. Deployment ####
# install.packages('rsconnect')
# library(rsconnect)
# Account setup
# rsconnect::setAccountInfo(name='tongjizzq',
#                          token='97EAC9318BC84543E690F971042956A9',
#                          secret='c2BjqD6RIF/1708hDvrPEGvyoCrp/CXYa4tumuRK')

# Deploy app
# rsconnect::deployApp(' ')
