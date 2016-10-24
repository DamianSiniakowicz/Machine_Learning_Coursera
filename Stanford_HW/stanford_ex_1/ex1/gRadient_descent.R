# Goal : implement gradient descent for uni and multivariate regression

# univar

# Step 1: get regression coefficients using built-in lm

food_truck_data <- read.csv(file = "ex1data1.txt", header = FALSE, col.names = c("Population", "Profit"), colClasses = c("numeric","numeric"))

food_truck_model <- summary(lm(data = food_truck_data, formula = Profit ~ Population))

food_truck_intercept <- food_truck_model$coefficients[1,1]
food_truck_population_coef <- food_truck_model$coefficients[2,1]

#plot(x = food_truck_data$Population, 
#     y = (food_truck_intercept + (food_truck_data$Population)*food_truck_population_coef), 
#     col= "red", type = "l")

#points(x = food_truck_data$Population, y = food_truck_data$Profit)

# Step 2 : get regression coefficient using gradient descent

Design_Matrix_Food <- matrix(data = c(rep(1,97),food_truck_data$Population),nrow = 97)
Parameter_Vector <- c(0,0)
Target_Vector <- food_truck_data$Profit

get_squared_error <- function(d_mat, p_vec, t_vec){
    predictions <- d_mat %*% p_vec
    error <- predictions - t_vec
    squared_error <- (error)^2
    sum(squared_error / (2*length(t_vec)))
}

descend_gradiently <- function(d_mat, p_vec, t_vec, alpha=.001, iters=10000){
    # d_mat : 97 by 2
    # p_vec : 2 by 1
    # t_vec : 97 by 1
    cost_history <- NULL
    for(i in 1:iters){
        cost_history <- c(cost_history, get_squared_error(d_mat,p_vec,t_vec))
        derivative_term <- t(alpha*(1/length(t_vec))*t(d_mat%*%p_vec - t_vec)%*%d_mat)
        p_vec <- (p_vec - derivative_term)
    }
    list(history = cost_history, parameters = p_vec)
}

#food_truck_grad_descent <- descend_gradiently(Design_Matrix_Food,Parameter_Vector,Target_Vector)

#lines(x = food_truck_data$Population, y = food_truck_data$Population * food_truck_grad_descent$parameters[2,1] + food_truck_grad_descent$parameters[1,1])

# close enough
#system_error <- get_squared_error(Design_Matrix_Food,c(food_truck_intercept,food_truck_population_coef), Target_Vector)
#Damian_error <- get_squared_error(Design_Matrix_Food, c(food_truck_grad_descent$parameters[1,1],food_truck_grad_descent$parameters[2,1]),Target_Vector)


# multi-var
housing_data <- read.csv(file="ex1data2.txt",header=FALSE, col.names = c("Area","Bedrooms","Price"), colClasses = c("numeric","integer","numeric"))
housing_model <- summary(lm(data = housing_data, formula = Price ~ Area + Bedrooms))

housing_intercept <- housing_model$coefficients[1,1]
housing_area_coef <- housing_model$coefficients[2,1]
housing_bedroom_coef <- housing_model$coefficients[3,1]

normalized_area <- (housing_data$Area - mean(housing_data$Area))/sd(housing_data$Area)
normalized_price <- (housing_data$Price - mean(housing_data$Price))/sd(housing_data$Price)


house_design_matrix <- matrix(data = c(rep(1,47),normalized_area,housing_data$Bedrooms) , nrow = 47)
house_parameter <- c(0,0,0)
house_target <- normalized_price

housing_descent <- descend_gradiently(house_design_matrix,house_parameter,house_target)

plot(x = 1:10000, y = housing_descent$history)

# close enough :-)
predicted_prices_normalized <- housing_descent$parameters[1,1] + housing_descent$parameters[2,1] * normalized_area + housing_descent$parameters[3,1] * housing_data$Bedrooms
predicted_prices <- predicted_prices_normalized * sd(housing_data$Price) + mean(housing_data$Price)
Damian_Error <- ((predicted_prices - housing_data$Price)^2)/94
prediction_data <- data.frame(Area = housing_data$Area,Bedrooms = housing_data$Bedrooms)
Howss <- lm(data = housing_data, formula = Price ~ Area + Bedrooms)
comp_predict <- predict(Howss,prediction_data)
System_Error <- ((comp_predict - housing_data$Price)^2)/94
error_difference <- abs(sum(System_Error) - sum(Damian_Error))/sum(System_Error)
cat("error difference between my model and built-in model as a percentage of the built in\'s error \n",error_difference)

#closed form solution
HDM <- matrix(data = c(rep(1,47),housing_data$Area,housing_data$Bedrooms),nrow=47)
HP <- c(0,0,0)
HT <- housing_data$Price
solution <- solve(t(HDM)%*%HDM)%*%t(HDM)%*%HT