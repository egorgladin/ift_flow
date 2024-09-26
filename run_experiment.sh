echo "Which algorithm would you like to run? Enter a number:\n1 - MMD flow,\n2 - noisy MMD flow,\n3 - IFT particle GD,\n4 - IFT particle GD with KL weight update"
read a
if [[ "${a}" -eq 1 ]]
then
python ift_experiment 9001 10000 20 mmd None 0
elif [[ "${a}" -eq 2 ]]
then 
python ift_experiment 9001 10000 20 mmd 2 4000
elif [[ "${a}" -eq 3 ]]
then 
python ift_experiment 9001 10000 20 ift None 0 0.001 MMD
elif [[ "${a}" -eq 4 ]]
then 
python ift_experiment 9001 10000 20 ift None 0 1 KL
else
echo "Input should be one of the numbers 1, 2, 3, 4."
fi