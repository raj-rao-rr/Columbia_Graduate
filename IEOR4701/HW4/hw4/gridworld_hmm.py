import numpy as np
import numpy.typing as npt


class Gridworld_HMM:
    def __init__(self, size, epsilon: float = 0, walls: list = []):
        if walls:
            self.grid = np.zeros(size)
            for cell in walls:
                self.grid[cell] = 1
        else:
            self.grid = np.random.randint(2, size=size)

        self.init = ((1 - self.grid) / np.sum(self.grid)).flatten('F')

        self.epsilon = epsilon
        self.trans = self.initT()
        self.obs = self.initO()

    def neighbors(self, cell):
        i, j = cell
        M, N = self.grid.shape
        adjacent = [(i, j), (i, j + 1), (i + 1, j), (i, j - 1), (i - 1, j)]
        neighbors = []
        for a1, a2 in adjacent:
            if 0 <= a1 < M and 0 <= a2 < N and self.grid[a1, a2] == 0:
                neighbors.append((a1, a2))
        return neighbors


    """
    3.1 and 3.2. Transition and observation probabilities
    """

    def initT(self) -> np.ndarray:
        """
        Create and return nxn transition matrix, where n = size of grid.
        """
        
        T = np.zeros((self.grid.size, self.grid.size))
        rows, cols = self.grid.shape
        
        # Iterate through the Row x Col board
        for row in range(rows):
            for col in range(cols): 
                
                # Re-mapping the universe based on whether hit wall
                if self.grid[row, col] != 1:
                    X = self.neighbors(cell=(row, col))
                   
                    # if transitions we assign each value across 64 states
                    for neighbor in X:
                        T[16*row + col, 16*neighbor[0] + neighbor[1]] = 1 / len(X)
                else:
                    T[16*row + col, 16*row + col] = 1
                    
        return T

    
    def initO(self):
        """
        Create and return 16xn matrix of observation probabilities, where n = size of grid.
        """
        O = np.zeros((16, self.grid.size))
        rows, cols = self.grid.shape
        
        def nesw_val(current_state, neighbors):
            """Compute the correct observation"""
            bit_exp = [1, 1, 1, 1]
            
            # itterate to check the direction of travel
            for idx in neighbors:
                if idx != current_state:

                    if idx[1] < current_state[1]:
                        bit_exp[3] = 0   # West available 
                    elif idx[1] > current_state[1]:
                        bit_exp[1] = 0   # East available
                    elif idx[0] > current_state[0]:
                        bit_exp[2] = 0   # South available
                    elif idx[0] < current_state[0]:
                        bit_exp[0] = 0   # North available
                    
            bit_exp = ''.join([str(i) for i in bit_exp])
            return int(bit_exp, 2)
            
        
        S = [(row, col) for row in range(rows) for col in range(cols)]
        for i, state in enumerate(S): 
            X = self.neighbors(cell=(state[0], state[1]))
            
            if self.grid[state[0], state[1]] != 1:
                real_e = nesw_val(state, X)
                pseudo_e = np.arange(0, 16).tolist()
                
                # compute XOR operation and return binary count (py version <3.9)
                discrepancy = [bin(e ^ real_e).count('1') for e in pseudo_e]
                proba = [(1-self.epsilon)**(4-d)*self.epsilon**d for d in discrepancy]
                
                O[:, i] = proba
        
        return O
    

    """
    3.3 Inference: Forward, backward, filtering, smoothing
    """

    def forward(self, alpha: npt.ArrayLike, observation: int):
        """Perform one iteration of the forward algorithm.
        Args:
          alpha (np.ndarray): Current belief state.
          observation (int): Integer representation of bitstring observation.
        Returns:
          np.ndarray: Updated belief state.
        """
        # TODO
        alpha = alpha.dot(self.trans) * self.obs[observation, :]
        return alpha


    def backward(self, beta: npt.ArrayLike, observation: int):
        """Perform one iteration of the backward algorithm.
        Args:
          beta (np.ndarray): Current array of probabilities.
          observation (int): Integer representation of bitstring observation.
        Returns:
          np.ndarray: Updated array.
        """
        # TODO
        beta = (beta * self.obs[observation, :]).dot(self.trans.T)
        return beta


    def filtering(self, observations: list[int]):
        """Perform filtering over all observations.
        Args:
          observations (list[int]): List of integer observations.
        Returns:
          np.ndarray: Alpha vectors at each timestep.
          np.ndarray: Estimated belief state at each timestep.
        """
        # TODO
        
        alpha_vectors = np.zeros((len(observations), self.grid.size))
        norm_Xt = np.zeros((len(observations), self.grid.size))
        
        # initialize alpha-zero as equally-weighted vector
        temp_alpha = np.ones((1, self.grid.size)) / self.grid.size
        
        for idx, obs in enumerate(observations): 
            temp_alpha = self.forward(temp_alpha, obs)
            
            alpha_vectors[idx, :] = temp_alpha
            norm_Xt[idx, :] = temp_alpha / temp_alpha.sum()
        
        return alpha_vectors, norm_Xt
    

    def smoothing(self, observations: list[int]):
        """Perform smoothing over all observations.
        Args:
          observations (list[int]): List of integer observations.
        Returns:
          np.ndarray: Beta vectors at each timestep.
          np.ndarray: Smoothed belief state at each timestep.
        """
        # TODO
        beta_vectors = np.ones((len(observations), self.grid.size))
        new_norm_Xt = np.zeros((len(observations), self.grid.size))
        
        # call forward algorithm to filter
        alpha_vectors, _ = self.filtering(observations)
        temp_beta = np.ones((1, self.grid.size))
        
        # iterate backwards through observation states
        for idx, obs in enumerate(observations[::-1]):
            
            norm_Xt = alpha_vectors[-(idx+1), :] * temp_beta
            new_norm_Xt[-(idx+1), :] = norm_Xt / norm_Xt.sum()
            
            temp_beta = self.backward(temp_beta, obs)
            beta_vectors[idx, :] = temp_beta
            
        return beta_vectors, new_norm_Xt


    """
    3.4. Parameter learning: Baum-Welch
    """

    def baum_welch(self, observations: list[int]):
        """Learn observation probabilities using the Baum-Welch algorithm.
        Updates self.obs in place.
        Args:
          observations (list[int]): List of integer observations.
        Returns:
          np.ndarray: Learned 16xn matrix of observation probabilities, where n = size of grid.
          list[float]: List of data likelihoods at each iteration.
        """
        M, N = self.grid.shape
        self.obs = np.ones((16, M * N)) / 16
        likelihoods = []

        while True:
            # compute our beta, alpha, and gamma values to be used in our Baum update, where betas are computed from the backward algorithm, 
            # the alphas are computed from the forward algorithm, and gammas are normalized vectors from the element wise produce of alpha and beta 
            betas, gammas = self.smoothing(observations)
            a1 = self.forward(self.init, observations[0])

            # iterate through each observation and sequentially add our gammas as observed, this will help to provided new
            # conditional estimates for our observation probabilites P(e|X)
            O = np.zeros((16, M * N))
            for i in range(len(observations)):
                O[observations[i]] += gammas[i]

            # normalize our observation and assign this matrix self.obs as our new observations probabilites
            # comptue the likelihood function (error term) which will help in determing convergence of the algorithm
            self.obs = np.nan_to_num(O / np.sum(O, axis=0))
            likelihoods.append(np.log(np.sum(a1 * betas[0])))

            # finally we denote conditions by which we converge to a solution and reach a locla maximum, 
            # namely if we've generated likelihoods whose value differs by less than 0.001 
            if len(likelihoods) > 1 and likelihoods[-1] - likelihoods[-2] < 1e-3:
                return self.obs, likelihoods
