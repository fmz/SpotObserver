//
// Created by fmz on 7/24/2025.
//

#include "spot-observer.h"

#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <ROBOT_IP> <username> <password>" << std::endl;
        return 1;
    }

    std::string robot_ip = argv[1];
    std::string username = argv[2];
    std::string password = argv[3];

    std::cout << "Connecting to Spot robot at " << robot_ip << " with user " << username << std::endl;

    int32_t spot_id = SOb_ConnectToSpot(robot_ip.c_str(), username.c_str(), password.c_str());

    bool ret = SOb_ReadCameraFeeds(spot_id, 0x3f);
    if (!ret) {
        std::cerr << "Failed to start reading camera feeds" << std::endl;
        SOb_DisconnectFromSpot(spot_id);
        return 1;
    }

    SOb_DisconnectFromSpot(spot_id);

    return 0;
}