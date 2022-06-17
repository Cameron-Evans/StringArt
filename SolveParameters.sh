for numOfEdgeNails in 200 250 300
do
    for numOfFieldNails in 10 20 30 40 50
    do
        for minDistanceBetweenNails in 6 8
        do
            for threadLen in 700 1000 1300
            do
                for neighborExclusion in 3 5 7
                do
                    for minDistanceFNailFromEdge in 5 10 15
                    do
                        for proportionOfThreadInk in 1.0 0.75 0.5
                        do
                            python3 main.py $numOfEdgeNails $numOfFieldNails $minDistanceBetweenNails $threadLen $neighborExclusion $minDistanceFNailFromEdge $proportionOfThreadInk
                        done
                    done
                done
            done
        done
    done
done
