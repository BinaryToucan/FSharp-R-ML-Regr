FROM mcr.microsoft.com/dotnet/runtime:5.0 AS base
WORKDIR /app

FROM mcr.microsoft.com/dotnet/sdk:5.0 AS build
WORKDIR /src
COPY ["F_R_ML/F_R_ML.fsproj", "F_R_ML/"]
RUN dotnet restore "F_R_ML/F_R_ML.fsproj"
COPY . .
WORKDIR "/src/F_R_ML"
RUN dotnet build "F_R_ML.fsproj" -c Release -o /app/build

FROM build AS publish
RUN dotnet publish "F_R_ML.fsproj" -c Release -o /app/publish

FROM base AS final
WORKDIR /app
COPY --from=publish /app/publish .
ENTRYPOINT ["dotnet", "F_R_ML.dll"]
