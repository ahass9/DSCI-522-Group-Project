# Binary Classification of Hotel Cancellations

## Authors: Ashifa Hassam, Nour Shawky, Athul Sasidharan, Kin Chung Choy

### Project Summary:

In this project we worked with an open hotel booking dataset containing reservations for both city and resort hotels and we selected a smaller set of features that would realistically be available at the time of booking, such as lead time, whether the guest is a repeat customer, previous cancellations, booking changes, special requests, reserved room type and deposit type. After a brief exploratory analysis, we confirmed that cancellations are fairly common and that patterns differ across hotel types and booking conditions, especially for repeated guests and deposit policies. We then framed the problem as a binary classification task: predict whether a booking will be cancelled or not. The data was split into training and test sets, and we built pipelines that combined preprocessing with three models: K-Nearest Neighbours, a decision tree and an RBF-kernel SVM.

Using 5-fold cross validation with F1 score on the validation set, KNN gave the best performance, so we chose it as our final model and evaluated it on the unseen test data. On the test set, our KNN model performs well enough with an accuracy of 73% and that it could plausibly support day-to-day hotel operations, at least as a first-pass tool for flagging higher-risk bookings. The confusion matrix shows that the model correctly identifies most non-cancellations and captures a reasonable portion of true cancellations, even though it still misses some risky bookings. That said, we do not see it as a finished solution. There is clear value in extending the feature set, tuning the model more carefully, and examining in detail which types of reservations are consistently misclassified. This would not only improve raw performance, but also give hotels a better sense of when to rely on the model’s predictions and when they should be treated as one input alongside staff experience and existing booking policies.

### Usage (how to run data analysis):

#### First-time setup

Run the following command from the root of the DSCI-522-Group-Project Repository:

```bash
conda-lock install --name dsci522_project conda-lock.yml
```

To then run the analysis notebook in jupyter, run the following command in the root of the repository:

```bash
jupyter lab
```

Open ```analysis.ipynb``` and navigate to the Kernel, then select ```Python[conda env:dsci522_project]```.

After that, in the menu options, select Restart Kernel and Run all Cells.

### Dependencies:
- ```python=3.12.12```
- ```pandas=2.3.3```
- ```conda-lock=3.0.4```
- ```altair=6.0.0```
- ```scikit-learn=1.7.2```
- ```matplotlib=3.10.8```
- ```ipykernel=7.1.0```
- ```vl-convert-python=1.8.0```

### Licenses:

The code for this project is licensed under the terms of the MIT Licence, offered under the [MIT open source license](https://opensource.org/license/MIT). The project report is licensed under [Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) License](https://creativecommons.org/licenses/by-nc-nd/4.0/).See the [LICENSE.md](https://github.com/ahass9/DSCI-522-Group-Project/blob/main/LICENSE.md) file for more information.
