
# 1.自定义SHAP可视化函数（适配CatBoost模型）####

library(shiny)         # 用于构建交互式网页应用
library(catboost)      # 用于加载CatBoost模型和预测
library(ggplot2)       # 用于数据可视化
library(shapviz)       # 用于SHAP值可视化
library(kernelshap)    # 用于计算SHAP值（核方法）
library(bslib)         # 用于自定义主题和样式

# 加载预训练模型（核心文件）
setwd("C:\\Users\\26553\\Desktop\\社区数据")
cat_final <- readRDS("cat_final.rds")

shapviz.catboost.Model <- function(object, X_pred, X = X_pred, collapse = NULL, ...) {
  if (!requireNamespace("catboost", quietly = TRUE)) {
    stop("未安装'catboost'包，请先安装")
  }
  stopifnot(
    "X必须是矩阵或数据框，不能是catboost.Pool类对象" =
      is.matrix(X) || is.data.frame(X),
    "X_pred必须是矩阵、数据框或catboost.Pool对象" =
      is.matrix(X_pred) || is.data.frame(X_pred) || inherits(X_pred, "catboost.Pool"),
    "X_pred必须包含列名" = !is.null(colnames(X_pred))
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

# 2.定义用户界面（UI）####
ui <- fluidPage(
  # 整体样式设置
  theme = bslib::bs_theme(
    bg = "white", 
    fg = "#333333", 
    primary = "#2c3e50",
    base_font = font_google("Lato")  # 使用Google Fonts提升字体美观度
  ),
  tags$head(
    tags$style(HTML("
      /* 标题样式 */
      .main-title {
        color: #2c3e50;
        font-weight: bold;
        text-align: center;
        margin: 20px 0 30px;
        font-size: 28px;
        text-shadow: 1px 1px 3px rgba(255,255,255,0.8);
      }
      /* 说明区域样式 */
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
      /* 输入面板样式 */
      .sidebar-panel {
        background-color: #ffffea;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        padding: 20px;
      }
      /* 按钮样式 */
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
      /* 结果区域样式 */
      .results-panel {
        margin-top: 20px;
      }
      /* 预测结果区域样式 */
      .prediction-results {
        background-color: #fbe0e0;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        padding: 25px;
        margin-bottom: 20px;
      }
      /* SHAP图区域样式 */
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
      /* 输入框标签样式 */
      .shiny-input-container > label {
        color: #2c3e50;
        font-weight: 600;
        margin-bottom: 5px;
        display: block;
      }
      /* 预测结果加粗 */
      .pred-result strong {
        color: #2980b9;
      }
    "))
  ),
  
  # 应用标题
  tags$h1(class = "main-title", "Predicting CKD in Community-Dwelling Older Adults with High Glycemic Status"),
  
  # 计算器说明（英文）
  div(class = "description-box",
      tags$h3(class = "description-title", "About This Calculator"),
      p(class = "description-text", 
        "This calculator uses the CatBoost machine learning model to assess the risk of chronic kidney disease (CKD) based on eight key indicators, trained on extensive patient data."),
      p(class = "description-text", 
        "Note: For reference only. Please consult a healthcare professional for diagnosis and treatment.")
  ),
  
  # 侧边栏布局（左侧输入，右侧输出）
  sidebarLayout(
    # 侧边栏面板（输入区域）
    sidebarPanel(class = "sidebar-panel",
                 # 输入字段（变量名+单位）
                 numericInput("Age", "Age (years)", value = 60, min = 1, max = 120),
                 numericInput("SBP", "Systolic Blood Pressure (mmHg)", value = 120, min = 50, max = 250),
                 numericInput("HbA1c", "Glycated Hemoglobin (%)", value = 6.5, min = 3, max = 20),
                 numericInput("Scr", "Serum Creatinine (μmol/L)", value = 65, min = 20, max = 200),
                 numericInput("UA", "Uric Acid (μmol/L)", value = 327, min = 100, max = 1000),
                 numericInput("Hb", "Hemoglobin (g/L)", value = 140, min = 50, max = 250),
                 numericInput("TG", "Triglycerides (mmol/L)", value = 1.5, min = 0, max = 20),
                 numericInput("BUN", "Blood Urea Nitrogen (mmol/L)", value = 5.5, min = 0, max = 50),
                 
                 # 预测按钮（点击触发预测）
                 actionButton("predictBtn", "Predict CKD Risk", class = "prediction-btn")
    ),
    
    # 主面板（输出区域）
    mainPanel(class = "results-panel",
              # 预测结果显示区域
              div(class = "prediction-results",
                  h3(class = "result-title", "Prediction Results (Threshold: 25%)"),
                  htmlOutput("predictionResult")
              ),
              
              # SHAP力图显示区域
              div(class = "shap-plot",
                  h3(class = "result-title", "SHAP Feature Contribution Plot"),
                  plotOutput("shapForcePlot", height = "400px")
              )
    )
  )
)

# 3.定义服务器逻辑（后台计算）####
server <- function(input, output) {
  # 定义预测函数
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
  
  # 监听预测按钮点击事件
  observeEvent(input$predictBtn, {
    prediction <- predictCKD()
    
    # 输出预测结果（使用HTML格式）
    output$predictionResult <- renderUI({
      HTML(
        paste0(
          "CKD Prediction Probability: <strong>", round(prediction$probability * 100, 2), "%</strong><br>",
          "Prediction Result: <strong>", prediction$label, "</strong>"
        )
      )
    })
    
    # 生成并输出SHAP特征贡献图
    output$shapForcePlot <- renderPlot({
      shp <- shapviz.catboost.Model(cat_final, X_pred = prediction$data)
      
      p_lt <- sv_force(shp, row_id = 1) +
        theme_minimal() +  # 使用更简洁的主题
        scale_fill_manual(values = c("#ff8696", "#bbe6dd")) +  # 更鲜明的对比色
        labs(title = "Feature Contribution to CKD Risk Prediction") +
        theme(
          plot.title = element_text(hjust = 0.5, size = 14, face = "bold", color = "#2c3e50"),
          axis.title = element_text(face = "bold", size = 11, color = "#34495e"),
          plot.margin = unit(c(0.8, 0.8, 0.8, 0.8), "cm"),
          panel.grid.major = element_line(color = "#ecf0f1", size = 0.5),
          legend.position = "none"  # 移除图例，使图表更简洁
        )
      
      print(p_lt)
    })
  })
}

# 4.运行应用程序shinyApp ####
shinyApp(ui = ui, server = server)


# 5.部署 ####
# install.packages('rsconnect')
library(rsconnect)





 # 设置工作目录
 setwd("C:\\Users\\26553\\Desktop\\社区数据\\代码\\网页计算器")  # 例如：setwd("C:/projects/ckd-calculator")
 
 
 
 
 
 # 检查是否安装了 catboost
 installed.packages()["catboost", ]
 
 # 彻底卸载 catboost
 if ("catboost" %in% installed.packages()) {
   remove.packages("catboost")
 }
 
 # 确认已卸载
 if (!"catboost" %in% installed.packages()) {
   message("catboost 已成功卸载")
 } else {
   stop("catboost 卸载失败，请手动删除安装目录")
 }
 
 
 # 加载 rsconnect 并配置账户
 library(rsconnect)
 rsconnect::setAccountInfo(
   name = "tongjizzq",
   token = "97EAC9318BC84543E690F971042956A9",
   secret = "c2BjqD6RIF/1708hDvrPEGvyoCrp/CXYa4tumuRK"
 )
 
 # 禁用 renv 检测
 Sys.setenv(RSCONNECT_RENV_ENABLED = "false")
 
 # 重新部署
 rsconnect::deployApp(
   appDir = getwd(),
   appName = "ckd-prediction-calculator",
   appFiles = c("app.R", "xgb_model.rds"),
   forceUpdate = TRUE,
   launch.browser = FALSE
 )


 
 
 
 
 
 
 
 # 加载必要的包（保留核心依赖，移除 catboost 的自动安装逻辑）
 required_packages <- c(
   "shiny", "bslib"
 )
 
 # 安装并加载其他核心包
 for (pkg in required_packages) {
   if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
     install.packages(pkg, repos = "https://cloud.r-project.org", quiet = TRUE)
   }
 }
 lapply(required_packages, library, character.only = TRUE, quietly = TRUE)
 
 # 手动加载本地 catboost 包（关键步骤）
 catboost_path <- file.path(getwd(), "catboost")  # 指向项目中的 catboost 文件夹
 if (dir.exists(catboost_path)) {
   # 将本地包路径添加到库路径
   .libPaths(c(catboost_path, .libPaths()))
   # 强制加载本地 catboost
   if (!require("catboost", character.only = TRUE, lib.loc = catboost_path)) {
     stop("本地 catboost 包加载失败，请检查文件夹是否正确")
   }
 } else {
   stop("未找到 catboost 文件夹，请确认已将解压后的文件夹放入项目目录")
 }
 

 
 library(rsconnect)
 rsconnect::setAccountInfo(
   name = "tongjizzq",
   token = "97EAC9318BC84543E690F971042956A9",
   secret = "c2BjqD6RIF/1708hDvrPEGvyoCrp/CXYa4tumuRK"
 )
 
 # 禁用 renv，强制上传本地 catboost 文件夹
 Sys.setenv(RSCONNECT_RENV_ENABLED = "false")
 
 rsconnect::deployApp(
   appDir = getwd(),
   appName = "ckd-prediction-calculator",
   # 明确包含 catboost 文件夹和其他必要文件
   appFiles = c("app.R", "xgb_model.rds", "catboost"),
   forceUpdate = TRUE,
   launch.browser = FALSE
 )

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 # 设置工作目录
 setwd("C:\\Users\\26553\\Desktop\\社区数据\\代码\\网页计算器")  # 例如：setwd("C:/projects/ckd-calculator")
 
 
 # 加载必要的包（仅保留核心依赖）
 required_packages <- c("shiny", "bslib")
 
 # 安装并加载其他包
 for (pkg in required_packages) {
   if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
     install.packages(pkg, repos = "https://cloud.r-project.org", quiet = TRUE)
   }
 }
 lapply(required_packages, library, character.only = TRUE, quietly = TRUE)
 
 # 强制加载项目中的 Linux 版 catboost 包（关键步骤）
 catboost_path <- file.path(getwd(), "catboost")  # 指向项目中的 catboost 文件夹
 if (dir.exists(catboost_path)) {
   .libPaths(c(catboost_path, .libPaths()))  # 添加本地包路径
   if (!require("catboost", character.only = TRUE, lib.loc = catboost_path)) {
     stop("Linux 版 catboost 包加载失败，请检查文件夹结构")
   }
 } else {
   stop("未找到 catboost 文件夹，请确认已放入项目目录")
 }
 
 # 加载核心模型（CatBoost 模型）
 cat_final <- readRDS("cat_final.rds")
 
 # UI 部分（仅保留预测输入和结果展示）
 ui <- fluidPage(
   theme = bslib::bs_theme(bg = "white", fg = "#333333"),
   tags$h1("CKD Risk Prediction (CatBoost)"),
   sidebarLayout(
     sidebarPanel(
       numericInput("Age", "Age (years)", value = 60, min = 1, max = 120),
       numericInput("SBP", "Systolic Blood Pressure (mmHg)", value = 120, min = 50, max = 250),
       numericInput("HbA1c", "Glycated Hemoglobin (%)", value = 6.5, min = 3, max = 20),
       numericInput("Scr", "Serum Creatinine (μmol/L)", value = 65, min = 20, max = 200),
       numericInput("UA", "Uric Acid (μmol/L)", value = 327, min = 100, max = 1000),
       numericInput("Hb", "Hemoglobin (g/L)", value = 140, min = 50, max = 250),
       numericInput("TG", "Triglycerides (mmol/L)", value = 1.5, min = 0, max = 20),
       numericInput("BUN", "Blood Urea Nitrogen (mmol/L)", value = 5.5, min = 0, max = 50),
       actionButton("predictBtn", "Predict CKD Risk")
     ),
     mainPanel(
       h3("Prediction Results"),
       verbatimTextOutput("predictionResult")
     )
   )
 )
 
 # Server 部分（核心预测逻辑）
 server <- function(input, output) {
   observeEvent(input$predictBtn, {
     # 构建输入数据
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
     
     # 转换为 CatBoost 所需的 Pool 格式
     new_pool <- catboost::catboost.load_pool(data = new_data)
     
     # 预测
     pred_prob <- catboost::catboost.predict(
       model = cat_final,
       pool = new_pool,
       prediction_type = "Probability"
     )
     
     # 结果判断
     pred_label <- ifelse(pred_prob > 0.2, "CKD Positive", "CKD Negative")
     
     # 输出结果
     output$predictionResult <- renderPrint({
       list(
         "CKD Probability" = paste0(round(pred_prob * 100, 2), "%"),
         "Prediction" = pred_label
       )
     })
   })
 }
 
 shinyApp(ui = ui, server = server)
 
 
 
 
 
 
 
 
 library(rsconnect)
 rsconnect::setAccountInfo(
   name = "tongjizzq",
   token = "97EAC9318BC84543E690F971042956A9",
   secret = "c2BjqD6RIF/1708hDvrPEGvyoCrp/CXYa4tumuRK"
 )
 
 # 禁用 renv 并强制部署
 # 强制上传所有文件，包括 catboost 子文件夹
 Sys.setenv(RSCONNECT_RENV_ENABLED = "false")
 rsconnect::deployApp(
   appDir = getwd(),
   appName = "ckd-prediction-calculator",
   # 显式列出 catboost 文件夹下的所有文件（确保完整上传）
   appFiles = c(
     "app.R", "cat_final.rds",
     list.files("catboost", recursive = TRUE, full.names = TRUE)
   ),
   forceUpdate = TRUE,
   lint = FALSE
 )
 

 # 安装本地 Linux 版 catboost 包
 # 注意：即使在 Windows 环境中，也需安装此 Linux 包用于后续上传到服务器
 install.packages(
   pkgs = "C:/Users/26553/Downloads/catboost-R-Linux-1.2.8.tgz",  # 本地包路径
   repos = NULL,  # 禁用 CRAN 仓库，强制本地安装
   type = "source"  # 以源码方式安装（忽略系统差异）
 )
 #renv::dependencies()
 