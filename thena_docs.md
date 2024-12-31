# Page: [THENA Overview](https://docs.thena.fi/thena)

THENA Overview
The trading hub and liquidity layer built on BNB Chain and opBNB. The THENA ecosystem encompasses a comprehensive suite of products and services, including:

- **THENA**: A Spot DEX where users can swap, acquire digital assets, and generate passive income.
- **ALPHA**: A Perpetuals DEX offering trading on over 270 crypto pairs with leverage up to 60x.
- **ARENA**: A social platform for trading competitions that provides a gamified experience for users and growth hacking tools for THENA's partners.
- **WARP**: (Launchpad – Upcoming)

## The Vision
The vision of THENA is to become the “SuperApp” platform set to onboard the masses, on-chain, with a CEX-grade experience. THENA has a versatile approach as we envision to become the most modular liquidity layer ever built to corner any type of liquidity needs from our Partners: Stablecoins, LSTs, Tokenized RWAs, Memecoins, AI tokens, etc.

## User Mass Onboarding
We have built our platform with the goal of simplifying the onboarding process and enhancing the experience for users of all levels, from beginners to the most experienced. Our ultimate aim is to become the SuperApp of DeFi, offering a highly decentralized alternative to Binance, where users can benefit from:

- **Wallet Abstraction**: Simplifies the onboarding process.
- **FIAT On/Off Ramp**: Facilitates initial purchases.
- **Debit Card and Banking Services**: Upcoming features to integrate traditional banking with DeFi.
- **Cross-Chain Bridge Functionalities**: Enables seamless asset transfers across different blockchains.
- **Digital Asset Acquisition**: Instant access to spot market trades.
- **Self-Custodial Earn Service**: Includes single staking of key assets such as BTC, ETH, BNB, and USDT.
- **Crypto Derivatives Trading**: Trade over 270 crypto assets with up to 60x leverage.
- **Permissionless Trading Competitions**: Upcoming feature to monetize trading activities.
- **Exclusive Access to Launchpools**: Upcoming opportunities for early investments in new projects.

## Building the Most Modular Liquidity Layer for Partners
### Incentives
The ve(3,3) tokenomics structure offers unparalleled flexibility in managing on-chain liquidity. Through weekly deposits in our open marketplace for incentives, partners can adjust the amount of token emissions their liquidity pools receive according to their specific needs.

###How to vote?

There are several ways to optimize your voting experience:
- If you are a liquidity provider on a pool yourself you might plan to increase the emissions to that pool and use your votes for that while still earning fees and bribes
- If you plan to acquire a specific coin anyways vote for the pool that has this coin as reward token. The platform usually averages nicely so you will usually not do much worse than average voting apr.
- If you want to squeeze out the optimum wait for the last moment and find the pools with the highest aprs. Make sure to not self dilute. This strategy is the most risky one since other voters might play the same game and you could get diluted.
- Play it safe by spreading your vote over a number of large pools where you can be almost certain to not be diluted.



### AMM Offering
We provide a diverse array of AMM models, including:
- Concentrated Liquidity AMM
- Classic UniV2-AMM
- 80/20 Balancer-style AMM
- Curve-style Stable-AMM

### AMM Modularity
The integration of Algebra Integral allows for a granular customization of our AMMs with the implementation of various plugins.

### Liquidity Management
Our Automated Liquidity Managers (ALM) partners offer a wide range of strategies to address specific liquidity management needs.

## What is THENA?
Watch here.

---

# Page: [THE Onboarding](https://docs.thena.fi/thena/the-onboarding)

## THE Onboarding
How to connect your Binance wallet to THENA

---

# Page: [THE Spot DEX](https://docs.thena.fi/thena/the-spot-dex)

## THE Spot DEX
Previous: THE Onboarding  
Next: Swap Guide

---



# Page: [Swap Guide](https://docs.thena.fi/thena/the-spot-dex/swap-guide)

## Swap Guide
How to execute your first transaction in THENA

---

# Page: [Limit Order](https://docs.thena.fi/thena/the-spot-dex/limit-order)

## Limit Order
### Introduction to Limit Orders
A limit order is a tool from CeFi in which users can buy or sell assets at a specified price or better, instead of relying on the market price at the time of execution. In a limit order, while the price is guaranteed, the order being executed is not - limit orders will be executed only if the price meets the order qualifications.

THENA has integrated the dLIMIT protocol, powered by Orbs, that brings this order type to DeFi in a decentralized manner. Users can use this tool to create decentralized limit orders by following the directions below.


### Abacus and livethe

Abacus serves as a yield optimization layer within ve(3,3) protocols like Thena, aiming to enhance ecosystem performance and provide seamless integration for users. It employs a liquid wrapper token system that unlocks veTOKENs, offering additional yield while building its own vePOSITION for voting purposes. Abacus holds veNFT positions across participating decentralized exchanges (DEXes) and utilizes them to vote on pools based on carefully tuned optimization routines. These routines prioritize pools that maximize profits for all participants. 
ABACUS DOCUMENTATION
MEDIUM

The rewards earned through veNFT votes are converted into blue-chip tokens. Subsequently, Abacus executes a Time-Weighted Average Purchase (TWAP) to either buy abcTOKENs from the market or lock DEX tokens. This process avoids slippage, prevents bot front-running, and ensures the effectiveness of each buyback. 
MEDIUM

liveTHE is one of Abacus's liquid wrappers built on top of Thena. It allows users to stake their THE tokens and receive liveTHE tokens in return, enabling them to participate in the Thena ecosystem without directly locking their tokens. This approach provides liquidity and flexibility to users while still capturing the benefits of staking. 
TWITTER

In summary, Abacus's vote optimizer enhances the efficiency and profitability of ve(3,3) protocols like Thena by strategically voting on pools and managing rewards. The liveTHE liquid wrapper offers users a flexible and liquid staking option within the Thena ecosystem.


### How to set-up Limit Orders?
**Prerequisites**:
- Ensure you have a Web3-enabled browser and sufficient balance in your wallet to cover the order amount and gas fees.

1. **Navigate to Limit**: Select ‘Limit’ from [thena.fi/swap](https://thena.fi/swap)
2. **Select Token**: Choose the token pair you wish to make a swap on. As an example, we will swap from WBNB to THE.
3. **Enter token swap quantity**: Enter the amount of token you would like to swap.
4. **Limit Price**: Type the highest price for trades in your order. Trades will ONLY be executed when the limit price you set is lower than or equal to the spot price.
5. **Place Order**: Click ‘Place Order’
6. **Confirmation**: After checking the trading details and the disclaimer, click ‘Confirm Order’ at the bottom of the column.
7. **Approve transaction**: Confirm the transaction from your wallet.

Once your transaction is approved, your Limit order is now waiting to be executed in the background. You can view the order history and progress on the page.

---

# Page: [TWAP Order](https://docs.thena.fi/thena/the-spot-dex/twap-order)

## TWAP Order
Introduction to dTWAP

---

# Page: [CROSS-CHAIN](https://docs.thena.fi/thena/the-spot-dex/cross-chain)

## CROSS-CHAIN
Interoperability powered by Axelar and Squid Router

At the heart of THENA's Cross-Chain Swap lies the innovative combination of Axelar's foundational infrastructure and Squid Router's application layer. Axelar serves as a powerful blockchain "internet infrastructure," ensuring seamless and secure cross-chain communication. In tandem, Squid Router, leveraging Axelar's prowess, facilitates efficient asset transfers and swaps across chains with a single click. This service enables the exchange of assets across a myriad of blockchains.

Whether you're looking to swap assets or bridge different blockchain ecosystems, THENA's Cross-Chain Swap, empowered by Axelar and Squid Router, promises unmatched efficiency and security.

---

# Page: [THE Tokenomics](https://docs.thena.fi/thena/the-tokenomics)

## THE Tokenomics
Introduction to THE tokenomics

A self-optimizing decentralized exchange and a liquidity layer, owned by protocols and the community. A grand design with tokenomics that seek to inspire the best behavior from all participants for the greater good of all stakeholders.

---

# Page: [Tokens (THE, veTHE, theNFT)](https://docs.thena.fi/thena/the-tokenomics/tokens-the-vethe-thenft)

## Tokens (THE, veTHE, theNFT)
Definition of THE token economy

---

# Page: [Initial Supply and Emissions Schedule](https://docs.thena.fi/thena/the-tokenomics/initial-supply-and-emissions-schedule)

## Initial Supply and Emissions Schedule
Breakdown of initial supply and token inflation

---

# Page: [veTHE Specs](https://docs.thena.fi/thena/the-tokenomics/vethe-specs)

## veTHE Specs
Lock your $THE into veTHE to start accessing the revenue of THENA

### veTHE Utility
- **Protocol revenue access**: veTHE holders can vote for gauges on a weekly basis, and access 90% of the trading fees and 100% of the voting incentives for the associated pool.
- **Governance participation**: veTHE holders can partake in governance and cast votes for the protocol improvement proposals.

### veTHE Voters Receive:
- Trading fees generated by the pool(s) they vote for.
- Voting incentives deposited for the pools they vote for.
- Weekly veTHE rebase (early adopter anti-dilution mechanics).

### veTHE Specifications
- **ve(3,3) Mechanics**: The Olympus DAO anti-dilution method, commonly known as the rebase mechanism, is combined with Curve's vote-escrowed model in the Solidly-initiated ve(3,3) Mechanics concept. To safeguard veTHE holders from dilution and to enable a dynamic distribution of veTHE among participants over time, the anti-dilution level has been capped at 30%.
- **Gauge**: A pool with dynamic $THE rewards based on veTHE weekly voting allocation. No negative voting.
- **Voting incentives**: Custom amount of tokens deposited by a protocol on a gauge to veTHE holders in exchange for their votes.
- **Max Lock**: 2 years.
- **Farming Boost**: This feature has not been included to prevent the emergence of any profit-maxi protocols on top of THENA. Conversely, a dynamic and decentralized governance over $THE emissions is fostered over time.
- **Flexibility**: veTHE positions can be merged, split, and sold on the secondary market.

### Voting
For voting, you need to be aware of epochs. Each epoch lasts for 7 days, after which the voting incentives and trading fees are distributed. You earn only from the gauges (pools) you have voted for.

- Trading fees and voting incentives are claimable as a lump sum after the next Epoch has ended (n+2).
- You have to vote weekly in order to be eligible for the fees and voting incentives, unless you use an optimizer.
- You can pre-approve your vote for a number of weeks in advance (coming soon).
- You can change or reset your vote at any time.
- Vote weights reset each Epoch. You need to vote every Epoch in order to earn the voting incentives and trading fees.

---

# Page: [veTHE Guide](https://docs.thena.fi/thena/the-tokenomics/vethe-guide)

## veTHE Guide
How to create a veTHE position, How to vote and earn, Voting FAQ and How to collect rewards

### How to create a veTHE position?
1. **Navigate to Lock**: From ‘Dashboard’, navigate to ‘Lock’ and click on ‘Get veTHE’. Alternatively, you can follow this link: [Lock veTHE](https://thena.fi/dashboard/lock).
2. **Create a Lock**: You will now see this on your screen.
3. **Select Amount and Duration**: Choose how much $THE to lock and for how long. Longer durations grant more voting power. A higher voting power in turn grants you a larger share of revenue.
4. **Confirm Lock**: Click on ‘Create New Lock’ and sign the series of transactions with your wallet.
5. **After the transactions are complete**, you can view your new veTHE position under ‘Dashboard’ → ‘Lock’.

### How to vote and earn with veTHE?
1. **Navigate to vote**: You can do so by heading to ‘Dashboard’ --> ‘Vote’. Alternatively, you can follow this link: [Vote veTHE](https://thena.fi/dashboard/vote).
2. **Choose Pool(s)**: You can view the different pools with their corresponding reward amount and voting APR. You can hover over the rewards to see the tokens you will receive rewards in if you vote for the pool. Note that reward amount and voting APR is not final until the end of the Epoch.
3. **Vote**: You can vote for multiple pools with one veTHE position. For demonstration purposes, we will split our votes between live THE/THE and SOL/BNB pool in a 77%/23% split. Once you have allocated your votes, click vote to proceed.
4. **Confirm vote**: Sign the series of transactions with your wallet.

Congratulations, you have completed your first weekly voting! Remember to vote again next Epoch!

### Voting FAQ
- **How should I choose the best pool to vote for?**
  - While there is no clear guideline on which pools are the best, here are some factors you can consider when choosing the best pool to vote for:
    - **APR (Annual Percentage Rate)**: A primary factor is the APR offered by the pool. Generally, a higher APR translates to greater rewards, assuming other factors remain constant.
    - **Reward tokens**: Since you can only claim your voting rewards after the next Epoch ends, the value of your rewards might fluctuate within this period. Risk-averse voters might find voting for stablecoin or major cryptocurrency pools more attractive, placing less emphasis on the pool’s APR.
    - **Dollar-Cost Averaging (DCA) Strategy**: If you aim to accumulate a specific cryptocurrency, consider voting for pools that align with this strategy. For instance, if you're accumulating DOGE, consistently voting for a DOGE/BNB pool ensures a steady inflow of DOGE as rewards.
    - **Total Rewards in Pool**: Consider the total rewards available in the pool and the number of votes it receives. For those with significant voting power or when voting in pools with limited total rewards, it’s crucial to consider the potential dilution effect from your votes. Voting in pools with a larger reward pool may mitigate this dilution.

- **When is the best time to vote?**
  - The best time to vote is near the end of the Epoch. This is when the rewards estimate is the most accurate and most protocols have deposited their voting incentives. As such, the voting APR is the most accurate and you can make the best choices! However, to avoid missing the voting period, you can also choose to vote earlier.

- **Why should I extend my lock?**
  - Your voting power correlates with the remaining duration of your token lock. Consider a hypothetical scenario where you earned $500 from voting with a one-year lock. Had you extended your lock to maintain the two-year duration, your voting power would have remained higher, and you could have earned more.

- **How do I minimize my gas fees used?**
  - We will soon be introducing cross-chain voting, where you can vote for opBNB and other upcoming networks with almost zero gas fees, allowing those with smaller veTHE positions to take part in weekly voting effectively. You can also accumulate rewards from many voting Epochs to collect at once in order to save on gas fees.

### How to collect rewards?
There are two parts of your rewards – rebase and the rewards that you voted for.

- **Rebase rewards**: You can claim your rebase immediately after the current Epoch ends (n+1 Epochs).
- **Voting rewards**: You can claim your voting rewards after the next Epoch ends (n+2 Epochs).

---

# Page: [ALPHA: THE Perpetual DEX](https://docs.thena.fi/thena/alpha-the-perpetual-dex)

## ALPHA: THE Perpetual DEX
Introduction

---

# Page: [Key Concepts](https://docs.thena.fi/thena/alpha-the-perpetual-dex/key-concepts)

## Key Concepts
Perpetuals

Assets traded on ALPHA are on-chain Perpetuals, a specific type of bilateral contracts that never expire. At the closing of the position, a cash-settlement payment is exchanged between holders of the two sides of the contracts, long and short. The payment is equal to the difference between the underlying asset price at the opening and at the closing of the trade, according to the leverage agreed between the two sides.

### Collateral & Cross-Margin Account
To start sending and accepting Intents and ensure their solvency within Derivatives contracts, users are required to deposit collateral ($USDT) into their cross-margin account. Compared to Order Book based DEXs, cross-margin accounts are a risk management tool that provide improved liquidity, financial flexibility and unnecessary liquidation of positions achieved by reduced margin requirements. Through this unique approach, ALPHA is acting as the first decentralized and trustless clearing house on the BNB chain.

Traditionally in trading, a margin refers to an amount of money borrowed to a broker to purchase a financial instrument. As the trades are executed with leverage in ALPHA, the unrealized gains of one of the two parties can quickly exceed the amount deposited as collateral in its own account. This situation translates into an ongoing loan by the other party, though its own margin account.

One particularity of ALPHA is the cross-margin nature of the capital employed as collateral. The position margins of a user are aggregated altogether in order to offset positive and negative unrealized gains, all with the same collateral. It not only enhances the trading experience of users, but also allows for an unmatched capital efficiency of the collateral deposited.

### Funding Rates
Funding rates are periodic payments between traders holding perpetual contract positions. Since perpetual futures contracts have no expiration date and can be held indefinitely, a price discrepancy can happen between the perpetual contract prices and the spot price of the underlying cryptocurrency.

When the funding rate is negative, shorts compensate for longs, and when it's positive, longs pay shorts. In a bullish market, the funding rate tends to be positive and increase, while in a bearish market, the opposite occurs.

Since Hedgers might be willing to hedge themselves on third party entities, such as CEXs, that are likely to charge them funding rates, which are to be implemented to the system at a later date.

---

# Page: [Trade on ALPHA](https://docs.thena.fi/thena/alpha-the-perpetual-dex/trade-on-alpha)

## Trade on ALPHA
THE user interface walk-through

---

# Page: [Hedgers](https://docs.thena.fi/thena/alpha-the-perpetual-dex/hedgers)

## Hedgers
Plug into SYMMIO and start filling intents from DeFi users

---

# Page: [Charts by TradingView](https://docs.thena.fi/thena/alpha-the-perpetual-dex/charts-by-tradingview)

## Charts by TradingView

---

# Page: [ALPHA Trading Guide](https://docs.thena.fi/thena/alpha-the-perpetual-dex/alpha-trading-guide)

## ALPHA Trading Guide
How to Trade on alpha.thena.fi

---

# Page: [theNFT Collection](https://docs.thena.fi/thena/thenft-collection)

## theNFT Collection
The founders of THENA theNFTs are a non-dilutive collection of only 1,734. Originally minted by THENA early adopters to bootstrap the ecosystem, each one grants a non-dilutive access to the swap fees generated by THENA and a dedicated role within the community.

### Benefits
Entitles holders to a 10% share of the trading fees from THENA. The royalty charged from secondary sales is 3%, 2% goes to a pool to which the original minters have a claim to — forever, while 1% goes to the NFT staking pool. The fees and royalties are distributed to a staking pool. If your theNFT is listed for sale, it cannot be staked, and will thus not earn fees.

### Secondary Marketplaces
theNFTs have been listed on 3 different NFT marketplaces:
- Element Market - the main marketplace
- tofuNFT - good UX
- NFTKEY - rarity ranking

### Token Address
theNFT: `0x2Af749593978CB79Ed11B9959cD82FD128BA4f8d`

---

# Page: [THE Open Marketplace for Liquidity](https://docs.thena.fi/thena/the-open-marketplace-for-liquidity)

## THE Open Marketplace for Liquidity
The Open Marketplace for Liquidity is the cornerstone of THENA veTHE gauges voting system serves as an open marketplace for liquidity, offering protocols the ability to execute tailor-made liquidity management strategies through voting incentives deposit. This unique approach ensures that protocols can bootstrap, scale, and maintain an adequate level of liquidity according to their specific needs. Additionally, THENA operates as a self-optimizing DEX, directing liquidity towards the most productive pools based on the fees they generate.

### Bootstrapping Liquidity
Protocols can solve the initial cold-start of liquidity by depositing voting incentives.

### Liquidity Management
Protocols can adjust their liquidity levels by modifying their weekly incentives deposit, enabling them to grow and maintain their pools in line with their strategic objectives.

### Self-optimizing Liquidity Allocation
THENA's ve33 tokenomics structure directs liquidity towards the most productive pools based on trading volumes and fees, ensuring efficient use of liquidity across the ecosystem. THENA’s marketplace for liquidity is designed to be both flexible and efficient, enabling protocols to manage their liquidity effectively while ensuring that users benefit from a robust and adaptive trading environment.

### Voting Incentives Specifications
- **Voting incentive claim**: veTHE holders receive rewards after the following Epoch in the form of a claimable lump sum.
- **Epoch duration**: 1 week (Thursday to Thursday).
- **Tokens**: Any whitelisted tokens can be deposited as voting incentive.
- **Voting incentive deposits**: Any time during the epoch. To attract votes from veTHE holders, voting incentive must be deposited at least a few hours before the Epoch changes at 00:00 UTC Wednesday-Thursday night.
- **Earning voting incentives (and trading fees)** requires weekly voting.

---

# Page: [Whitelist Tokens](https://docs.thena.fi/thena/the-open-marketplace-for-liquidity/whitelist-tokens)

## Whitelist Tokens
Get tokens whitelisted on THENA

New tokens need to be whitelisted. The team keeps control over new tokens being listed in order to filter out scams and illegitimate projects. Update: Tax tokens are now compatible.

### How to whitelist a token?
Contact the team through Discord or Twitter.

---

# Page: [Add Gauges](https://docs.thena.fi/thena/the-open-marketplace-for-liquidity/add-gauges)

## Add Gauges
Anyone can add a gauge for a pool without holding veTHE. Adding a gauge will allow voting incentives to be deposited for the given pool.

### How to add a gauge?
1. **Connect Wallet**
2. **Navigate to "More" -> "Protocols"**
3. **Select "Add Gauge"**
4. **Select your pool and confirm**

---

# Page: [Add Voting Incentives](https://docs.thena.fi/thena/the-open-marketplace-for-liquidity/add-voting-incentives)

## Add Voting Incentives
Anyone can add voting incentives to a pool for which a gauge has been added. The voting incentive must be deposited in tokens that have been whitelisted. The more voting incentives added for a gauge, the more $THE emissions it is likely to receive in the next epoch.

### How to add voting incentives?
1. **Connect Wallet**
2. **Navigate to "More" -> "Protocols"**
3. **Select "Add Voting incentives"**
4. **Select your gauge, the amount of tokens for the Epoch, and confirm**
5. **Repeat the process every Epoch**

---

# Page: [THE Liquidity Pools](https://docs.thena.fi/thena/the-liquidity-pools)

## THE Liquidity Pools
THENA offers a variety of liquidity pools adapted to a wide combination of asset pairs and strategies. Our innovative pools combine Concentrated Liquidity AMM, dynamic fee structures, and seamless integration with our Automated Liquidity Manager (ALM) partners for an enhanced user experience and capital efficiency. From Classic UniV2 AMM and Curve-Like Stable AMM, to innovative FUSION pools and soon Balancer-like 80-20 weighted pools, they enable protocols to achieve their liquidity objectives.

---

# Page: [Introduction to FUSION](https://docs.thena.fi/thena/the-liquidity-pools/introduction-to-fusion)

## Introduction to FUSION
FUSION is a collaborative development that abstracts and automates the management of concentrated liquidity pools, enhancing capital efficiency and user experience.

### Concept of Concentrated Liquidity
In the classic Uniswap V2 AMM, liquidity is evenly distributed across the entire price curve of the token pair (0, +∞), equally rewarding dormant liquidity and liquidity effectively utilized for end-user swaps. DEX utility tokens emitted to rent LP liquidity are not capital efficient, and fees cannot be directed towards liquidity actively used for swaps.

Concentrated liquidity pools enable the discrete allocation of the liquidity within specific price ranges, allowing them to collect trading fees when the current price falls within the established range. Similar to an order book, the aggregated liquidity exhibits a more "concentrated" distribution around the market price of the token pair. For end-users, concentrated liquidity provides better price execution than UniV2 pools for any given level of TVL, while more efficiently rewarding LPs.

### Algebra - Concentrated Liquidity AMM Provider
Algebra provides the underlying concentrated liquidity AMM technology for FUSION that enables the liquidity provision through custom price ranges, facilitating discrete liquidity provision and the execution of advanced market-making operations on THENA.

### Dynamic Fee Structure
Algebra includes an innovative fee structure composed of:
- **Base fee**: Base component that can be freely set and modified by the core team without needing to redeploy liquidity to a new pool. This feature offers an extra layer of flexibility for THENA's protocol partners when executing their liquidity strategies on the BNB chain.
- **Dynamic fee**: Component that automatically adjusts according to market volatility, liquidity, and trading volume to maximize generated fees. During periods of high volatility, fees should be increased to compensate for potential losses faced by liquidity providers. Conversely, when trading volume is low and there is sufficient liquidity, fees should be reduced to attract more volume.

### Impermanent Loss Risk
In concentrated liquidity pools, passive LPs are more subject to impermanent losses. As the price shifts, LPs end up holding more of one asset while traders acquire more of the other. LPs may eventually hold a single asset position as the price of the pair approaches either the lower or upper limit of the range. The situation can worsen when the price surpasses one of the boundaries: the LP position now consists of a single asset that is decreasing in value and not generating trading fees.

Given the inherent volatility of the crypto market, LP positions in concentrated liquidity pools need frequent rebalancing to avoid extreme impermanent losses. Although concentrated liquidity pools bring a higher level of granularity to DeFi, profiting from liquidity provision within custom ranges demands sophisticated market-making skills that may be beyond the reach of the average retail liquidity provider.

### Active Liquidity Managers ("ALM") Integration
The integration of concentrated liquidity pools with various ALM addresses the complexity of these pools for non-professional liquidity providers. An ALM protocol actively adjusts LP ranges to maximize the fee generation while mitigating the impermanent loss risk. ALMs are natively integrated into FUSION, providing LPs with a seamless experience and access to a battle-tested market-making service.

ALM partners integrated into FUSION:
- Gamma
- DefiEdge
- ICHI

---

# Page: [Liquidity Pools Typology](https://docs.thena.fi/thena/the-liquidity-pools/liquidity-pools-typology)

## Liquidity Pools Typology
### FUSION (Automated Concentrated Liquidity AMM)
- **Gamma - Narrow**: The narrow range strategy focuses on a tight price range between two assets. This strategy works well for assets with high and positive price correlation as it concentrates the price range where most trading activity occurs. The Narrow Range also maximizes the fee-earning potential during periods of low market volatility. However, in the long term, higher fees may not compensate for the higher impermanent loss risk.
- **Fee structure**: Dynamic

### Gamma - Wide
The wide-range strategy involves distributing liquidity over an extensive price range between two assets. This approach is especially effective for assets with low price correlation and substantial price fluctuations. The strategy generates consistent fees during periods of market volatility, especially when high token volatility is expected, such as after a key release or unforeseen news. In the long run, the savings in impermanent loss will likely outweigh the higher fees associated with a narrow range.
- **Fee structure**: Dynamic

### Gamma - Stable
Similar to the v1 Stable Pool, this pool strategy is specifically designed for assets expected to consistently trade at near parity, such as stablecoins or synthetics. The FUSION Stable Pool allows for improved price execution compared to the v1 Stable Pool, resulting in higher volumes and fees generation. However, in case of a depeg between the paired assets, the pool could become entirely imbalanced and consist of only one asset, making it impossible to rebalance the pool. This extreme situation would not occur with the V1 stable AMM.
- **Fee structure**: Dynamic

### Gamma - Correlated
The Correlated strategy distributes the liquidity in a tight range around an evolutive price peg between a yield accruing token and its underlying based asset. This pool not only addresses the Liquid Staked Derivatives (LSD) tokens that are accruing the staking rewards over time (e.g., $ETH-$frxETH or $BNB-$ankrBNB), but also any other tokens integrated into a yield-generating market.
- **Fee structure**: Dynamic

### DefiEdge
DefiEdge strategies concentrate the liquidity close to the market price of the pair, through multiple ranges. The level of concentration of the liquidity deployed by DefiEdge strategies are in between the Wide and Narrow strategies from Gamma, offering an alternative risk/reward LP opportunity. The strategy can be adjusted directly with the core, on demand.
- **Fee structure**: Dynamic

### ICHI
ICHI provides single token deposit strategies. LPs deposit a single token that the strategy pairs with another token to generate yield through automated liquidity management operations on THENA. The strategy actively accumulates the deposited token and generates trading fees. The deposited token always keeps a [65%-95%] weight within the pair, providing a directional edge to LPs. ICHI offers a wide range of custom strategies to help THENA's partners achieve liquidity goals such as treasury management, sell-side incentives, and tailored on-chain market-making operations.
- **Fee structure**: Dynamic

### Classic AMM
UniV2-style variable pool designed for uncorrelated pairs of assets such as $BNB-$USDC. v1 Variable Pool comprises two tokens that users can swap, with the exchange rate determined by each token's relative supply and demand through the constant product formula. These pools are ideal for trading more volatile tokens, as they can rapidly adapt to fluctuating market conditions.
- **Fee structure**: 0.2%

### Stable AMM
Curve style of stable pool for correlated or loosely pegged pair of assets, such as $FRAX-$BUSD or $USDT-$USDC. v1 Stable Pool is a type of liquidity pool that is designed specifically for assets that are expected to consistently trade at near parity, such as stablecoins or synthetics. Unlike traditional AMMs, which use a constant product formula to calculate prices, sAMMs use a constant sum formula. This helps to reduce slippage and price impact, making sAMMs particularly well-suited for stable assets.
- **Fee structure**: 0.01%

### Manual Concentrated Liquidity AMM
The manual range strategy allows LPs to set their preferred price range and adjust it in a highly customizable way. Custom price ranges set-up is best suited for skilled investors confident in their market-making skills. The Manual Range Pool allows Market Makers individuals or organizations to build on top of THENA and earn 97% of the swap fees generated by their operations. 3% of the swap fees are retained and further deposited as voting incentives on FUSION gauges as a revenue share to veTHE holders. Manual Range Pools are not included in the gauges system, which means they are not eligible for earning $THE farming rewards.
- **Fee structure**: Dynamic

---

# Page: [Earn THE](https://docs.thena.fi/thena/the-liquidity-pools/earn-the)

## Earn THE
Provide liquidity and stake your LP tokens to earn $THE. There are no deposit or withdrawal fees. You can withdraw and remove liquidity at any time.

### In which proportion should I provide my liquidity?
- **FUSION (Automated CLAMM)**: Tokens need to be added in a 50:50 ratio.
- **Classic AMM**: Tokens need to be added in a 50:50 ratio.
- **Stable AMM**: The UI will automatically suggest the correct ratio, which can be other than 50:50.
- **Manual CLAMM**: The UI will automatically suggest the correct ratio, which can be other than 50:50.

### How to provide liquidity to earn $THE?
Guide here.

---

# Page: [Earn Trading Fees](https://docs.thena.fi/thena/the-liquidity-pools/earn-trading-fees)

## Earn Trading Fees
You can provide liquidity without staking the LP tokens. In this case, you will earn all of the trading fees for your proportionate share of the pool. There are no deposit or withdrawal fees. You can remove liquidity at any time.

- **Classic AMM**: LPs earn 100% of the fees.
- **Manual CLAMM**: LPs earn 97% of the fees.
- **FUSION (Automated CLAMM)**: LPs do not earn anything if their positions are not staked.

### How to provide liquidity to earn trading fees?
Guide here.

---

# Page: [ARENA](https://docs.thena.fi/thena/arena)

## ARENA
Where SocialFi meets trading

ARENA is a unique Social Hub for trading competitions natively integrated into THENA’s ecosystem. In ARENA, whitelisted partners can create permissionless and fully-customizable trading competitions on both spot and perpetual markets to engage their community, create engagement around their tokens, and boost their brand awareness. It is a growth hacking tool that can be leveraged by our partners at multiple levels to improve their user acquisition & retention rate, increase their website traffic, and deepen their liquidity in THENA.

### ARENA has been designed to align the interest of three key stakeholders:
- **Traders**: GameFi: ARENA offers a gamified experience where they can earn rewards based on their trading performance. ARENA also has other SocialFi features, such as event promotion, notifications, a profile view where they can analyze their own and other users’ activity. They will also be able to mint and be identified by their own .thena IDs, instead of the classic 0x address, for a more personal user experience.
- **Partner projects**: Activity and deeper liquidity: Trading competitions generate engagement, trading volumes, and fees for a token. In THENA, emissions are governed by veTHE holders that vote for pools on a weekly basis, and capture a corresponding share of the fees and voting incentives deposited by protocols. As a result, the volumes and fees generated by trading competitions are an alternative way to attract more votes than the traditional “bribing” system to deepen the liquidity of a pool. Growth Hacking tool: Whitelisted partners will be able to set up their own trading competitions and relay them on socials to boost their reach. Running joint trading competitions between multiple partners enables some massive community crossover and helps raise each partner’s brand awareness. As each whitelisted partner will have their own profile page that includes hyperlinks towards their website and socials, they will be able to increase their platform traffic, social engagement, and ultimately improve their user acquisition rate.
- **KOLs**: Audience monetization: KOLs are able to create trading competitions and earn up to 25% of the prize pool raised. No initial funding is needed; all they have to do is share the trading competitions created on their socials to attract users and increase participation. Copy trading will be implemented in the future to enhance traders' KOLs experience in ARENA and facilitate onboarding their community. Community expansion: As ARENA's network effect increases with more Partners and Users being onboarded, KOLs will be able to leverage the platform as a user acquisition funnel to broaden their audience and increase their social engagements. Thanks to .thena IDs, KOLs could create trading competitions together and benefit from community crossover.

---

# Page: [THENA IDs](https://docs.thena.fi/thena/arena/thena-ids)

## THENA IDs
THENA IDs are your primary identity within the THENA Ecosystem through our own Naming Service. By owning a THENA ID, your wallet address will be replaced with your chosen name, all across ARENA and THENA’s products suite. This way you can stand out from the crowd.

### THENA IDs minting prices start at 5 USDT and vary according to the length of the name chosen. They can also be gifted.

### THENA IDs let you customize your THENA profile with the following:
- Displaying your own theNFT as a profile picture
- Setting a custom name color
- Adding our website link
- Linking your X profile
- Displaying your bio
- Suggesting your profile to others
- Choosing profile’s theme background

In the future, THENA ID holders will get special early access to new features and access to further profile customization options.

### How to Mint THENA IDs?
1. **Step 1**: Navigate to THENA ID
   - Go to the THENA website ([thena.fi](https://thena.fi))
   - Click on "Arena" in the website header.
   - Select "THENA ID" from the dropdown menu. Alternatively, visit [thena.fi/arena/id](https://thena.fi/arena/id).
2. **Step 2**: Mint Your THENA ID
   - Click on "Mint".
   - Choose your unique THENA ID. Note: The shorter the length of your ID, the more USDT is required to mint it.
   - Once you have decided on a suitable ID, click "Mint Now". Remember, each unique THENA ID can only be minted once and is non-fungible. It cannot be changed once minted.
3. **Step 3**: Approve the Transactions
   - Approve the transactions with your wallet.
4. **Step 4**: View Your Minted THENA ID
   - Congratulations! You have now minted your unique THENA ID. View your ID under "Profile". Your ID will also be displayed whenever you use any of THENA’s ecosystem products.

---

# Page: [Trading Competition Guide](https://docs.thena.fi/thena/arena/trading-competition-guide)

## Trading Competition Guide
How to create a trading competition

---

# Page: [THE Roadmap](https://docs.thena.fi/thena/the-roadmap)

## THE Roadmap

---

# Page: [Security](https://docs.thena.fi/thena/security)

## Security
### Audits
THENA V1 has been audited by PeckShield.

THENA V2 has been audited by OpenZeppelin.

THENA is protected by Aegis from Lossless. THENA was originally adapted from the Velodrome codebase, which is directly derived from the Solidly smart contracts that have been open-sourced in March 2022.

The AMM part of Solidly has been audited by PeckShield that revealed 5 low-severity and 1 informal findings. There have been no security-related incidents involving Solidly smart contracts since their deployment on Fantom in February 2022. The Velodrome codebase went through a security audit and a peer review as part of a Code4rena bug bounty contest. All high or medium risk issues were either resolved pre-deployment, except for one known issue (users can claim eligible rewards from ExternalBribe contracts more than once) that has been addressed via a wrapped contract solution.

### Multisig
All transactions will have to go through a multisig.
- **Setup**: 4/6 signers needed for a transaction, 3 from THENA, 3 publicly known entities from Ankr, GrizzlyFi, and DEUS Finance (0x7d70ee3774325c51e021af1f7987c214d2caa184)
- **Timelock**: 12 hours (0x5d7deb17be6c6243d6d65205b5293edceb676561)

---

# Page: [Collaboration](https://docs.thena.fi/thena/collaboration)

## Collaboration
Join THE Ecosystem

---

# Page: [Brand Assets](https://docs.thena.fi/thena/brand-assets)

## Brand Assets
Creating THENA related content? Look no further.

Manual Guide: Creating a Concentrated Liquidity (CL) Position on Thena
Step 1: Understanding Concentrated Liquidity (CL) in Thena
Concentrated Liquidity in Thena allows liquidity providers (LPs) to allocate their capital within a specific price range. This method allows you to earn fees when the price of the token pair is within that range, rather than providing liquidity across the entire price curve.

Fee Earnings: You earn fees when trades occur within your price range. The fees are automatically distributed based on the liquidity you provide and the trades made within your range.
Step 2: Prerequisites
Before setting up a Concentrated Liquidity position, ensure the following:

Ensure you are connected to Thena’s platform or any compatible decentralized exchange (DEX) supporting the Thena protocol.
Wallet: You need a Web3 wallet (e.g., MetaMask, Trust Wallet, or WalletConnect).
Tokens: Have the necessary tokens for the liquidity pair (e.g., ETH, USDT, DAI, etc.).
Gas Fees: Ensure you have enough ETH (or the native blockchain token) to cover gas fees for the transactions.
Step 3: Access the Thena DEX Platform
Navigate to the Thena DEX platform: Go to Thena's website or the appropriate platform hosting the Thena protocol.
Connect your wallet: Click on the Connect Wallet button and select your wallet provider (e.g., MetaMask, WalletConnect).
Step 4: Choose a Token Pair
Select the Token Pair: Choose the token pair you wish to provide liquidity to. For example, you might select ETH/USDT or DAI/USDC.
Ensure Token Availability: Make sure you have both tokens available in your wallet to match the liquidity pair. You'll need to provide both tokens in proportion to their value.
Step 5: Set Your Price Range
To create a Concentrated Liquidity (CL) Position, you'll define the price range within which your liquidity will be active.

Check Current Price: Look at the current price of the token pair on Thena.

Set the Price Range:

Lower Bound: The minimum price at which your liquidity will be active.
Upper Bound: The maximum price at which your liquidity will be active.
Example: If ETH/USDT is trading at $2,000, you could set a price range of $1,800 to $2,200. Your liquidity will only be used when the price is within this range.

Tip: Choose a price range based on your market expectations. If you expect the price to stay within a specific range, concentrate your liquidity within that range to maximize your fees.

Thena has dynamical fees due to algebra

Low Fee Tier (e.g., 0.05%): This is often used for highly liquid, stable pairs where there’s frequent trading.
Medium Fee Tier (e.g., 0.3%): Used for pairs with moderate volatility and liquidity.
High Fee Tier (e.g., 1%): Typically for pairs with high volatility or lower liquidity.
You will choose the appropriate fee tier based on the token pair and your strategy for earning fees. The higher the fee tier, the more fees you can potentially earn, but it might come with higher volatility and less frequent trades in some pairs.

Step 7: Add Liquidity
Once you’ve set the price range and chosen the fee tier, you can now provide liquidity to the Thena pool.

Go to the Liquidity Page: Navigate to the Provide Liquidity section on Thena's platform.
Enter the Liquidity Amount: Input the amount of each token (e.g., ETH and USDT) that you want to provide within the price range. The platform will calculate how much of each token is required to meet the selected pair's value.
Review the Price Range and Fee Tier: Double-check that your price range and fee tier selection are correct.
Add Liquidity: Click the Add Liquidity button.
Approve the Transaction: You will need to approve the transaction in your wallet and confirm the gas fee.
Once the transaction is confirmed, your liquidity will be added to the Thena pool within the price range you've defined.

Step 8: Monitor Your Position
After adding liquidity, it’s important to monitor the position:

Track the Current Price: Regularly check the current price of the token pair to ensure it remains within your specified price range. If the price moves outside your range, your liquidity will no longer be used.
Check Fee Earnings: Thena will automatically distribute fees based on trades within your price range. You can track the fees you've earned on the platform.
Adjust Your Position (Optional): If the price moves significantly outside your range, you may want to adjust your price range or add more liquidity to ensure that you remain a part of the active pool.
Step 9: Remove Liquidity
If you wish to exit your position or adjust your liquidity, you can remove it manually:

Navigate to "Remove Liquidity": In Thena’s interface, go to the section for removing liquidity.
Select the Amount of Liquidity: Choose how much liquidity you want to withdraw.
Confirm and Remove Liquidity: After reviewing your liquidity and fees earned, confirm the transaction in your wallet.

---

# Page: [Official Links](https://docs.thena.fi/thena/official-links)

## Official Links
- **Website**: [thena.fi](https://www.thena.fi)
- **Twitter**: [ThenaFi](https://twitter.com/ThenaFi)
- **Discord**: [Join Discord](https://discord.gg/thena)
- **Telegram**: [Thena_Fi](https://t.me/Thena_Fi)
- **Telegram Announcements**: [ThenaFi](https://t.me/ThenaFi)
- **Zealy Quests**: [Zealy](https://zealy.io/c/thenafi/)
- **Medium**: [ThenaFi Medium](https://medium.com/@ThenaFi)
- **LinkedIn**: [ThenaFi LinkedIn](https://www.linkedin.com/company/thenafi/)
- **Facebook**: [ThenaFi Facebook](https://www.facebook.com/profile.php?id=61555896159299)
- **Link3**: [ThenaFi Link3](https://link3.to/thena)
- **Github**: [ThenaFi Github](https://github.com/ThenafiBNB/THENA-Contracts)
- **Immunefi**: [Immunefi Bounty](https://immunefi.com/bounty/thena/theNFT%20Collection)
- **Dune Dashboards**: [Dune Dashboard 1](https://dune.com/0xkhmer/thena) | [Dune Dashboard 2](https://dune.com/impossiblefinance/thena)
- **BNB to opBNB Bridge**: [Bridge](https://thena.zkbridge.com)
- **List of smart contracts**: [Google Sheets](https://docs.google.com/spreadsheets/d/1nk-OLHL6yZ3rvTYo2-OXTVpjgfo5PVyufmM1U4S8geA/edit#gid=0)
- **Regional**:
  - **China**: [ThenaCNCIS](https://t.me/thenaCNCIS)
  - **Korea**: [ThenaKorea Twitter](https://twitter.com/ThenaKorea) | [ThenaKorea Telegram](https://t.me/THENAKRKorea) | [ThenaKorea Telegram](https://t.me/THENAKorea)
  - **Turkey**: [ThenaFi Turkey](https://t.me/ThenaFiTurkiye)

---