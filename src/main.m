clc
clear
close all



%% -------------------- Dataset -------------------------------------------
% Carregando a base de dados e os rótulos
dataset = load('dataset.dat');
rotulos = load('rotulos.dat' ); % 0 - Maligno   e  1 - Benigno
% Nomeando os atributos
nome_features = {'radius mean', 'texture mean','perimeter mean','area mean','smoothness mean', 'compactness mean','concavity mean','concave points mean','symmetry mean','fractal dimension mean','radius se','texture se','perimeter se','area se','smoothness se','compactness se','concavity se', 'concave points se', 'symmetry se', "fractal dimension se","radius worst","texture worst","perimeter worst","area worst","smoothness worst","compactness worst","concavity worst","concave points worst","symmetry worst","fractal dimension worst"};

% Normalizando a base de dados
dataset = normalizando_dataset(dataset);

%% -------------------- Analise da base de dados --------------------------

% Plotando o histograma das distribuições dos atributos
[simetria,curtose]=histogramaplot(dataset, nome_features);
figure

% Plotando a correlação entre os atributos
heatmap(corr(dataset))

%% -------------------- Treinamento e validação ---------------------------
% Definindo o k folds do modelo de treinamento
k = 10;

% Dividindo a base de dados em treino e teste por meio método do k-fold
[index_teste,index_treino] = K_fold(dataset, k);

% Vetor que armazena a acurácia de cada Modelo
acuracia_bayseando = zeros(k,1);
acuracia_LDA = zeros(k,1);



for i=1:1:k
    % Pegando os dados de treino e teste da rodada 
    [dataset_teste, rotulos_teste] = fold_rodada(index_teste(i,:),dataset,rotulos);
    [dataset_treino, rotulos_treino] = fold_rodada(index_treino(i,:),dataset,rotulos); 
    
    % Treinando o modelo Bayesiano com os dados de treino
    classe_bayesano = ClassificadorBayesano(dataset_treino,rotulos_treino,dataset_teste)';   
    
    % Treinando o classificador LDA com os dados de treino
    classe_LDA = ClassificadorLDA(dataset_treino,rotulos_treino,dataset_teste); 
    
    % Calculando a acurácia dos dos modelos:
    acuracia_LDA(i,1) = acuracia(rotulos_teste,classe_LDA);

    acuracia_bayseando(i,1) = acuracia(rotulos_teste,classe_bayesano);

end

disp("Acurácia do classificador byesiano: " + mean(acuracia_bayseando));
disp("Acurácia do classificador LDA: " + mean(acuracia_LDA));

clear index_teste index_treino acuracia_bayseando acuracia_LDA
clear i dataset_teste dataset_treino rotulos_treino classe_bayesano rotulos_teste


%% -------------------- Funções implementadas -----------------------------
% Função que plota o histograma das distribuições dos atributos
function [simetria,curtose] = histogramaplot(dataset, nome_features)
    % Calcula o tamanho da base de daods
    tam_dataset = size(dataset);
    % Vetor que armazena a simetria dos atributos
    simetria = zeros(2,tam_dataset(2));
    % Vetor que armazena a curtose dos atributos
    curtose = zeros(2,tam_dataset(2));
    % Plota as distribuições dos 15 primeiros atributos
    figure
    for i = 1:1: tam_dataset(2)/2
        % Plota a distribuição do atributo da primeira classe
        subplot(3,5,i)
        histogram(dataset(1:212,i), 'FaceColor', 'blue', 'EdgeColor', 'black');
        hold on;
        % Calcula a simetria e curtose da primeira classe
        simetria(1,i) = skewness(dataset(1:212,i));
        curtose(1,i) = kurtosis(dataset(1:212,i));
        % Plota a distribuição do atributo da segunda classe sobreposta com a primeira
        histogram(dataset(213:end,i), 'FaceColor', 'red', 'EdgeColor', 'black');
        legend('M','B');
        hold off;
        % Calcula a simetria e curtose da primeira classe
        simetria(2,i) = skewness(dataset(213:end,i));
        curtose(2,i) = kurtosis(dataset(213:end,i));
        xlabel(nome_features(i));
    end

     % Plota as distribuições dos 15 últimos atributos
    figure
    for i = (tam_dataset(2)/2 +1):1: tam_dataset(2)
        % Plota a distribuição do atributo da primeira classe
        subplot(3,5,i-tam_dataset(2)/2)
        histogram(dataset(1:212,i), 'FaceColor', 'blue', 'EdgeColor', 'black');
        hold on;        
        % Calcula a simetria e curtose da primeira classe
        simetria(1,i) = skewness(dataset(1:212,i));
        curtose(1,i) = kurtosis(dataset(1:212,i));        
        % Plota a distribuição do atributo da segunda classe sobreposta com a primeira
        histogram(dataset(213:end,i), 'FaceColor', 'red', 'EdgeColor', 'black');
        legend('M','B');
        hold off;
        % Calcula a simetria e curtose da primeira classe
        simetria(2,i) = skewness(dataset(213:end,i));
        curtose(2,i) = kurtosis(dataset(213:end,i));
        xlabel(nome_features(i));
    end
       
end

% Função que classifica a base de dados usando o classificador bayesano do caso geral
function classes_preditas = ClassificadorBayesano(dataset,rotulos,amostras)
    % Tamanho do rótulos
    tam_rotulos = size(rotulos);

    % Dividir a base de dados de acordo com os rotulos
    [dataA,rotuloA,dataB,rotuloB,tipo_classes] = baseDiv(dataset,rotulos);
    
    % Calcula a covariaça de cada classe 
    CovariancaA = cov(dataA);
    CovariancaB = cov(dataB);
    
    % calcula a média de cada classe
    mediaA = mean(dataA)';
    mediaB = mean(dataB)';
    
    % Probabilidade apriori de cada classe
    probApriori_A = length(rotuloA)/tam_rotulos(1);
    probApriori_B = length(rotuloB)/tam_rotulos(1);

    %classes_preditas = zeros(length(amostras));
    % Aplica o caso geral do classificador bayesiano calcula as probabilidades da amostra de teste ser de cada classe
    for i=1:1:length(amostras)
        prob = [];
        % Função de custo para o caso geral
        prob(1) = log(det(CovariancaA)) + ((amostras(i,:)' - mediaA)')*(inv(CovariancaA))*(amostras(i,:)' - mediaA) - 2*log(probApriori_A); 
        prob(2) = log(det(CovariancaB)) + ((amostras(i,:)' - mediaB)')*(inv(CovariancaB))*(amostras(i,:)' - mediaB) - 2*log(probApriori_B); 

        % Seleciona a classe que minimize a função de custo
        [~, ind] = min(prob);
        classes_preditas(i) = tipo_classes(ind);
    end
end

% Função que classifica a base de dados usando o LDA como classificador 
function classe = ClassificadorLDA(dataset,rotulos, amostras)
    % Dividir a base de dados de acordo com os rotulos
    [dataA,~,dataB,~,~] = baseDiv(dataset,rotulos);
    
    % Calcula a média de cada classe 
    mediaA = mean(dataA)';
    mediaB = mean(dataB)';
    
    % Calcula o vetor projeção w
    Sw = cov(dataA) + cov(dataB);
    Sb = (mediaA-mediaB)*(mediaA-mediaB)';
    invSw_SB = (inv(Sw))*Sb;
    
    % Calcula os autovalores e autovetores
    [V, ~] = eig(invSw_SB);
    
    % Define a rpojeção que mais maximeze os dados  
    w = V(:,1);
    classe = w;
    
    % Para classificar define o limiar por meio da projeção w dos atributos
    limiar = ((w')*mediaA + (w')*mediaB)/2;

    % Divide o base de dados de acordo com o limiar
    [ClasseMenor,ClasseMaior] = limiares(dataset,rotulos,limiar,w);
    
    % Calcula a projeção de cada amostra teste
    vetor_W = amostras*w;
    
    % Define qual a classe pertence as amostras teste
    for i=1:1:length(vetor_W)
        if vetor_W(i,:) < limiar
            classe(i,:) = ClasseMenor;
        else
            classe(i,:) = ClasseMaior;
        end
    end
end

% Função que divide a base de dados com base no limiar
function [RotuloMenor, RotuloMaior] = limiares(dataset,rotulos,limiar,w)
    % Tamanho da base de rótulos
    tam_rotulos = size(rotulos);
    % Variáveis auxiliares para o indices das base de dados
    indA = 1; indB = 1;
    % Calcula a projeção de cada amostra da base de dados
    W = dataset*w;
    % Difine os rotulos com base no limiar das projeções da base de dados
    for i=1:1:tam_rotulos(1)
       if(W(i,:)<limiar)
           rotuloA(indA,:) = rotulos(i);
           indA = indA + 1;
       else
           rotuloB(indB,:) = rotulos(i);
           indB = indB + 1;
       end
    end
    % Retorna os rótulos pertecente as amostras menores e maiores que o limiar
    RotuloMenor = mode(rotuloA);
    RotuloMaior = mode(rotuloB);
end

% Função que divide a base de dados e acordo com suas classes
function [dataA,rotuloA,dataB, rotuloB, tipo_classes] = baseDiv(dataset,rotulos)
    % Tamanho da base de dados
    tam_rotulos = size(rotulos);
    % Define quais os rótulos das amostras
    tipo_classes = unique(rotulos,'stable');
    % Váriaveis auxiliares para indicar os indices
    indA = 1; indB = 1;
    % divide a base de dados em duas por meio da suas classes
    for i=1:1:tam_rotulos(1)
       if(rotulos(i) == tipo_classes(1))
           dataA(indA,:) = dataset(i,:);
           rotuloA(indA,:) = rotulos(i);
           indA = indA + 1;
       else
           dataB(indB,:) = dataset(i,:);
           rotuloB(indB,:) = rotulos(i);
           indB = indB + 1;
       end
    end
end

% Função que dividide a base de dados entre treino e validação usando o metodo k-fold
function [teste, treino] = K_fold(data_set,K)
    % O método de cruzamento k-fold consiste em dividir o conjunto total
    % de dados em k subconjuntos mutualmente exclusivos do mesmo tamanho.
    
    % Para implementar o método, a ideia é criar um vetor que indique através 
    % dos indices quem vai ser os objetos de treino e quem vai ser os objetos 
    % de teste, dessa forma possibilitar dividir a base de dados entre
    % treino e teste.
    
    % A função K_fold retorna os indices de treino e os indices de teste.

    N_objetos = size(data_set); %Descobre quem é a quantidade de atributos 'M' e a quantidade de objetos 'N' 
    resto = mod(N_objetos,K);
    dataset_indx = randperm(N_objetos(1)); %cria um vetor aleatório com a permutação 1 atea quantidade de objetos sem repeti-los.
     if resto(1) ~= 0
        for i=1:1:resto
           dataset_indx(:,i) = [];
        end
     end
    quant_grupo = floor(N_objetos(1)/K); %Define a quantidade de elementos em cada subconjunto K-fold, arredondado 
    vetor_index_teste = zeros(K,quant_grupo);
    vetor_index_treino = zeros(K,length(dataset_indx)-quant_grupo);
    ind_inicial = 1; %variavel auxiliar que aponta para o primeiro indice do subconjunto k-fold
    ind_final = quant_grupo; %variavel auxilar que aponta para o ultimo indice do subconjunto k-fold

    for i=1:K

        %Constroi o vetor de indice para os testes
        vetor_index_teste(i,:) = dataset_indx(1,ind_inicial:ind_final);
        
        %Constroi o vetor de indices para os treinos
        if i == 1
            vetor_index_treino(i,:) = dataset_indx(1,(ind_final+1):end);
        elseif i == K
            vetor_index_treino(i,:) = dataset_indx(1,1:(ind_inicial-1));
        else
            vetor_index_treino(i,:) = [dataset_indx(1,1:(ind_inicial-1)), dataset_indx(1,(ind_final+1):end)];
        end
        ind_inicial = ind_final+1; %Indice inicial recebe o ultimo indice mais 1
        ind_final = ind_final+ quant_grupo; %Indice final recebe o final mais a quantidade de elementos do k-fold, já que o periodo se repete em cada k-fold
    end
    teste = vetor_index_teste;
    treino = vetor_index_treino;
end

% Função que seleciona a base de dados
function [atributos, classes]= fold_rodada(indices,features,rotulos)
    % A função dataset_rodada() retorna os atributos da base de dados conforme 
    % os index definidos pelo método k-fold. 
    % A função retorna um vetor contendo os atributos nas colunas e as 
    % amostras nas linhas.

    n = length(indices);
    tamanho_features = size(features);
    features_selecionados = zeros(n,tamanho_features(2)); %Cria a base de dados dos atributos selecionados
    rotulos_selecionados = zeros(n,1); %Cria a base de dados dos rotulos selecionados
    %Preenche os dados de acordo com a divisão de subgrupos
    for i=1:n
        features_selecionados(i,:) = features(indices(1,i),:);
        rotulos_selecionados(i,1) = rotulos(indices(1,i),1);
    end
    atributos = features_selecionados;
    classes = rotulos_selecionados;
end

% Função que calcula a porcetagem de acertos usando acuracia
function taxa_acerto = acuracia(rotulos_reais,rotulos_preditas)
    % Tanto rotulos_reais quanto rotulos_preditas são vetores onde os
    % que contem as classes definidas nas linhas e uma unica coluna.
    n_amostras = length(rotulos_reais);
    acertos = 0;

    for i=1:1:n_amostras
        if rotulos_reais(i,1) == rotulos_preditas(i,1)
            acertos = acertos + 1;
        end
    end
    taxa_acerto = (acertos/n_amostras)*100;
    
end

% Função que normaliza a base de dados usando o método f1score
function dados = normalizando_dataset(features)
    % Função que normaliza os atributos usando o metodo zscore, assim os
    % atributos devem ter média zero e desvio padrão um.

    [n_amostras, n_atributos] = size(features);

    dados_normalizados = zeros(n_amostras,n_atributos);

    for i=1:1:n_atributos
        dados_normalizados(:,i) = zscore(features(:,i));
    end
    dados = dados_normalizados;
end
